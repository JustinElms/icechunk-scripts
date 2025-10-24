import datetime
import itertools
import multiprocessing as mp
import glob
import time
import os
import shutil
import tempfile
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
from icechunk.xarray import to_icechunk
from icechunk import Repository, Session, local_filesystem_storage, s3_storage

N_PROCS = 4


class ic_manager:

    def __init__(self, dataset_key: str) -> None:
        self.configs_dir = "/home/ubuntu/icechunk/ic-configs"
        self.log_file = "ic_log.txt"

        self.dataset_key = dataset_key

        self.dataset_config = self.__read_config()

        storage_config = self.__get_storage()

        if not Repository.exists(storage_config):
            Repository.create(storage_config)

        self.repo = Repository.open(storage_config)

    def __get_storage(self) -> s3_storage:
        return s3_storage(
            bucket="icechunk",
            prefix=self.dataset_key,
            access_key_id="admin",
            secret_access_key="password",
            endpoint_url="http://142.130.249.35:9000",
            allow_http=True,
            force_path_style=True,
        )

    def get_repo_timestamps(self) -> np.ndarray:
        session = self.repo.readonly_session("main")
        with xr.open_zarr(session.store, consolidated=False, decode_times=False) as ds:
            timestamps = ds.time.values

        return timestamps

    def __read_config(self) -> dict:
        with open(f"{self.configs_dir}/{self.dataset_key}.json", "r") as f:
            config = json.load(f)
        return config

    def get_forecast_dates(self) -> np.ndarray:
        """
        Returns a list of all forecast dates in the dataset directory
        """
        file_path = self.dataset_config["data_path_template"].format(
            forecast_date="*", forecast_run="*", forecast_time="*", variable="*"
        )
        files = sorted(glob.glob(file_path))
        forecast_dates = [
            f.replace(self.dataset_config["data_path_root"], "").split("/")[0]
            for f in files
        ]

        return list(set(forecast_dates))

    def init_repo_data(self) -> None:
        """
        Gets an array of inputs containing timestamp, model_run, variable, file_path
        for each file to be written to the icechunk repository. If a timestamp is already
        present in the repository, the mode is set to 'r+' to overwrite the previous data.
        """
        forecast_dates = self.get_forecast_dates()
        fc_times = np.arange(
            0,
            self.dataset_config["frequency"] * (self.dataset_config["n_times"] + 1),
            self.dataset_config["frequency"],
        )

        nc_data = list(
            itertools.product(
                forecast_dates,
                self.dataset_config["model_runs"],
                [f"{ft:03d}" for ft in fc_times],
                [0],
                self.dataset_config["variables"],
                self.dataset_config["drop_vars"],
                ["a"],
                ["time"],
                [None],
            )
        )

        for idx, nd in enumerate(nc_data):
            nc_files = glob.glob(
                self.dataset_config["data_path_template"].format(
                    forecast_date=nd[0],
                    forecast_run=nd[1],
                    forecast_time=nd[2],
                    variable=nd[4],
                )
            )
            if len(nc_files) > 0:
                nc_data[idx] = [*nd, nc_files[0]]
            else:
                nc_data[idx] = [*nd, None]

        input_df = pd.DataFrame(
            nc_data,
            columns=[
                "forecast_date",
                "forecast_run",
                "forecast_time",
                "timestamp",
                "variable",
                "drop_vars",
                "mode",
                "append_dim",
                "time_idx",
                "file",
            ],
        )
        input_df.dropna(subset=["file"], inplace=True)
        input_df["forecast_date"] = pd.to_datetime(
            input_df["forecast_date"], format="%Y%m%d"
        )

        input_df["timestamp"] = (
            input_df["forecast_date"]
            + pd.to_timedelta(
                input_df["forecast_time"].astype(int).values, unit="hours"
            )
            - datetime.strptime(
                self.dataset_config["reference_time"], "%Y-%m-%d %H:%M:%S"
            )
        )
        if self.dataset_config["frequency"] != 24:
            input_df["timestamp"] = input_df["timestamp"] + pd.to_timedelta(
                input_df["forecast_run"].astype(int).values, unit="hours"
            )
        input_df["timestamp"] = input_df["timestamp"].dt.total_seconds().astype(int)
        input_df = input_df.sort_values(
            by=[
                "forecast_date",
                "forecast_run",
                "forecast_time",
            ]
        )
        input_df = input_df.drop_duplicates(
            subset=["timestamp", "variable"], keep="last"
        )

        # determine which timestamps need to be updated
        written_timestamps = self.get_repo_timestamps()
        for idx, ts in enumerate(written_timestamps):
            df_idx = input_df[input_df["timestamp"] == ts].index
            input_df.loc[df_idx, "mode"] = "r+"
            input_df.loc[df_idx, "append_dim"] = None
            input_df.loc[df_idx, "time_idx"] = idx

        input_df = input_df[input_df["mode"] != "r+"].reset_index(drop=True)

        input_df = input_df.drop(
            columns=[
                "forecast_date",
                "forecast_run",
                "forecast_time",
                "timestamp",
                "variable",
            ]
        )

        return input_df.values.tolist()

    @staticmethod
    def write_timestamp(args) -> None:
        drop_vars, mode, append_dim, time_idx, file, session = args
        with xr.open_dataset(file, decode_times=False).drop_vars(drop_vars) as ds:
            if mode == "r+":
                region = {
                    "time": slice(time_idx, time_idx+1),
                    "depth": slice(0, 51),
                    "latitude": slice(0, 850),
                    "longitude": slice(0, 1800),
                }
                to_icechunk(ds, session, mode=mode, region=region)
            else:
                to_icechunk(ds, session, mode=mode, append_dim=append_dim)
        return session


if __name__ == "__main__":

    mp.set_start_method("forkserver")
    manager = ic_manager("giops_day")
    inputs = manager.init_repo_data()

    input_chunks = np.array_split(inputs, np.ceil(len(inputs) / N_PROCS))
    t0 = time.time()
    for input_chunk in input_chunks:
        print(f"Processing files: \n{',\n'.join(input_chunk[:, -1].astype(str))}")

        session = manager.repo.writable_session("main")
        with ProcessPoolExecutor(max_workers=N_PROCS) as executor:
            # submit the writes
            fork = session.fork()
            input_chunk = np.append(
                input_chunk,
                np.repeat(fork, len(input_chunk)).reshape((len(input_chunk), 1)),
                axis=1,
            )
            remote_sessions = executor.map(manager.write_timestamp, input_chunk)

        session.merge(*remote_sessions)
        session.commit(f"Wrote files: \n{',\n'.join(input_chunk[:,-1].astype(str))}")
