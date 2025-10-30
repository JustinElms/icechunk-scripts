import itertools
import glob
import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
import zarr
from icechunk.xarray import to_icechunk
from icechunk import (
    Repository,
    RepositoryConfig,
    s3_storage,
    ManifestConfig,
    ManifestSplitCondition,
    ManifestSplittingConfig,
    ManifestSplitDimCondition,
)


class ic_manager:

    def __init__(self, dataset_key: str) -> None:
        self.configs_dir = "/home/ubuntu/icechunk/ic-configs"
        self.log_file = "ic_log.txt"

        self.dataset_key = dataset_key

        self.dataset_config = self.__read_config()

        storage_config = self.__get_storage()

        if not Repository.exists(storage_config):
            split_config = ManifestSplittingConfig.from_dict(
                {
                    ManifestSplitCondition.AnyArray(): {
                        ManifestSplitDimCondition.DimensionName("time"): int(
                            10 * 24 / self.dataset_config["frequency"]
                        )
                    }
                }
            )
            repo_config = RepositoryConfig(
                manifest=ManifestConfig(splitting=split_config),
            )
            self.repo = Repository.create(storage_config, config=repo_config)
            self.repo.save_config()
        else:
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
        try:
            with xr.open_zarr(
                session.store, consolidated=False, decode_times=False
            ) as ds:
                timestamps = ds.time.values
        except zarr.errors.GroupNotFoundError:
            timestamps = []
        return timestamps

    def __read_config(self) -> dict:
        with open(f"{self.configs_dir}/{self.dataset_key}.json", "r") as f:
            config = json.load(f)
        return config

    def get_forecast_dates(self, start_date: str) -> list:
        date = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(datetime.today(), "%Y%m%d")

        forecast_dates = []
        while date <= end:
            forecast_dates.append(date)
            date += timedelta(days=1)

        return [datetime.strftime(fd, "%Y%m%d") for fd in forecast_dates]

    def init_repo_data(self, start_date: str, end_date: str | None = None) -> None:
        """
        Gets an array of inputs containing timestamp, model_run, variable, file_path
        for each file to be written to the icechunk repository. If a timestamp is
        already present in the repository, the mode is set to 'r+' to overwrite the
        previous data. Only use for "best estimate" datasets.

        Args:
            start_date: the date of the initial forecast to be added to the repo
            end_date: the date of the last forecast to be added to the repo

        Returns:
            list: a list of inputs that can be passed to write_timestamp formatted
                for multiprocessing
        """
        forecast_dates = self.get_forecast_dates(start_date, end_date)
        fc_times = np.arange(
            0,
            self.dataset_config["end_hour"] + 1,
            self.dataset_config["frequency"],
        )

        input_df = list(
            itertools.product(
                [None],
                forecast_dates,
                self.dataset_config["model_runs"],
                [f"{ft:03d}" for ft in fc_times],
                [None],
                self.dataset_config["variables"],
                self.dataset_config["drop_vars"],
                ["a"],
                ["time"],
                [None],
                [None],
            )
        )

        input_df = pd.DataFrame(
            input_df,
            columns=[
                "timestamp",
                "forecast_date",
                "forecast_run",
                "forecast_time",
                "datetime",
                "variable",
                "drop_vars",
                "mode",
                "append_dim",
                "region",
                "file",
            ],
        )

        input_df["datetime"] = pd.to_datetime(
            input_df["forecast_date"], format="%Y%m%d"
        )

        input_df["datetime"] = input_df["datetime"] + pd.to_timedelta(
            input_df["forecast_time"].astype(int).values, unit="hours"
        )
        if self.dataset_config["frequency"] != 24:
            input_df["datetime"] = input_df["datetime"] + pd.to_timedelta(
                input_df["forecast_run"].astype(int).values, unit="hours"
            )
        input_df["timestamp"] = input_df["datetime"] - datetime.strptime(
            self.dataset_config["reference_time"], "%Y-%m-%d %H:%M:%S"
        )
        input_df["timestamp"] = input_df["timestamp"].dt.total_seconds().astype(int)

        input_df = input_df.sort_values(
            by=[
                "timestamp",
                "forecast_date",
                "forecast_run",
                "forecast_time",
            ]
        ).reset_index(drop=True)

        # only keep most recent timestamp for each variable
        for ts in input_df["timestamp"].unique():
            for var in self.dataset_config["variables"]:
                temp_df = input_df[
                    (input_df["timestamp"] == ts) & (input_df["variable"] == var)
                ].copy()
                temp_df = temp_df[::-1]

                for idx, row in temp_df.iterrows():
                    nc_files = glob.glob(
                        self.dataset_config["data_path_template"].format(
                            forecast_date=row["forecast_date"],
                            forecast_run=row["forecast_run"],
                            forecast_time=row["forecast_time"],
                            variable=row["variable"],
                        )
                    )

                    if len(nc_files) > 0:
                        temp_df.loc[idx, "file"] = nc_files[0]
                        break
                ts_idx = temp_df.index
                file_idx = temp_df["file"].last_valid_index()
                if file_idx is None:
                    file_idx = ts_idx[-1]
                else:
                    input_df.loc[file_idx, "file"] = nc_files[0]
                ts_idx = ts_idx[ts_idx != file_idx]
                input_df.drop(ts_idx, inplace=True)

        # remove timestamps with missing forecast data
        # TODO add logging for fcts with missing data
        counts = input_df.groupby(["forecast_date", "forecast_run", "forecast_time"])[
            "timestamp"
        ].transform("size")
        input_df = input_df[
            (counts == len(self.dataset_config["variables"]))
            & (input_df["file"].notnull())
        ]

        import pickle
        with open("giops_2d_df.pkl","wb") as f:
            pickle.dump(input_df,f)

        # determine which timestamps need to be updated
        written_timestamps = self.get_repo_timestamps()
        if len(written_timestamps) == 0:
            df_idx = input_df[
                (input_df["forecast_date"] == input_df["forecast_date"].min())
                & (input_df["forecast_run"] == input_df["forecast_run"].min())
                & (input_df["forecast_time"] == input_df["forecast_time"].min())
            ].index

            if all(
                input_df.loc[df_idx, "variable"].values
                == self.dataset_config["variables"]
            ):
                input_df.loc[df_idx, "append_dim"] = None
        else:
            for idx, ts in enumerate(written_timestamps):
                region = self.dataset_config["dims_size"]
                region = {key: slice(0, value) for key, value in region.items()}
                region["time"] = slice(idx, idx + 1)
                df_idx = input_df[input_df["timestamp"] == ts].index
                input_df.loc[df_idx, "mode"] = "r+"
                input_df.loc[df_idx, "append_dim"] = None
                for i in df_idx:
                    input_df.at[i, "region"] = region

        # Group output data by write type
        write_df = input_df.loc[
            (input_df["mode"] == "a") & (input_df["append_dim"].isnull())
        ]
        update_df = input_df.loc[(input_df["mode"] == "r+")]
        append_df = input_df.loc[
            (input_df["mode"] == "a") & (input_df["append_dim"].notnull())
        ]

        groups = [write_df, update_df, append_df]

        input_df = pd.concat(groups)

        inputs = []
        timestamps = input_df["timestamp"].unique()
        for ts in timestamps:
            temp_df = input_df.loc[input_df["timestamp"] == ts].drop(
                columns=[
                    "forecast_date",
                    "forecast_run",
                    "forecast_time",
                    "timestamp",
                    "variable",
                    "datetime",
                ]
            )
            inputs.append(temp_df.values.tolist())

        return inputs

    @staticmethod
    def write_timestamp(args) -> None:
        drop_vars, mode, append_dim, region, file, session = args
        with xr.open_dataset(file, decode_times=False).drop_vars(drop_vars) as ds:
            if mode == "r+":
                to_icechunk(ds, session, mode=mode, region=region)
            elif append_dim:
                to_icechunk(ds, session, mode=mode, append_dim=append_dim)
            else:
                to_icechunk(ds, session, mode=mode)
        return session
