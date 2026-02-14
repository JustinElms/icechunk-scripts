import glob
import json
import os
import warnings
from datetime import datetime

import cftime
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from obstore.store import LocalStore
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser, NetCDF3Parser
from virtualizarr.registry import ObjectStoreRegistry

from icechunk import (
    ManifestConfig,
    ManifestSplitCondition,
    ManifestSplittingConfig,
    ManifestSplitDimCondition,
    Repository,
    RepositoryConfig,
    VirtualChunkContainer,
    local_filesystem_store,
    s3_storage,
)

warnings.filterwarnings(
    "ignore",
    message="Numcodecs codecs are not in the Zarr version 3 specification*",
    category=UserWarning,
)
warnings.filterwarnings("ignore", category=zarr.errors.UnstableSpecificationWarning)

# TODO:
# - Add functionality to remove data (e.g. timestamps older than 2 yrs)
# - Add function to clean up datastore (remove unused files)
# - Add error handling and other user feedback
# - Add function to spot check data
# - Add verification of dataset and ic config file - raise errors for bad or missing
#   values


class IcechunkInterface:

    def __init__(self, dataset_key: str) -> None:
        self.ic_config = self.__read_config("ic_config.json")

        self.log_file = self.__configure_log_file(dataset_key)

        self.dataset_config = self.__read_config(
            f"{self.ic_config.get("dataset_configs_dir")}/{dataset_key}.json"
        )

        storage_config = self.__get_storage(dataset_key)

        if not Repository.exists(storage_config):
            split_config = ManifestSplittingConfig.from_dict(
                {
                    ManifestSplitCondition.AnyArray(): {
                        ManifestSplitDimCondition.DimensionName("time"): 1
                    }
                }
            )
            repo_config = RepositoryConfig(
                manifest=ManifestConfig(splitting=split_config),
            )
            virtual_container = VirtualChunkContainer(
                url_prefix="file:///data/", store=local_filesystem_store(path="/data/")
            )
            repo_config = RepositoryConfig()
            repo_config.set_virtual_chunk_container(virtual_container)
            self.repo = Repository.create(storage_config, config=repo_config)
            self.repo.save_config()
        else:
            self.repo = Repository.open(
                storage_config, authorize_virtual_chunk_access={"file:///data/": None}
            )

        self.initialized = len(list(self.repo.ancestry(branch="main"))) > 1

    def __get_storage(self, dataset_key) -> s3_storage:
        s3_config = self.ic_config.get("s3_config")
        return s3_storage(
            bucket=s3_config.get("bucket"),
            prefix=dataset_key,
            region="us-east-1",
            access_key_id=s3_config.get("user"),
            secret_access_key=s3_config.get("password"),
            endpoint_url=s3_config.get("url"),
            allow_http=True,
            force_path_style=True,
        )

    def __read_config(self, path: str) -> dict:
        with open(path, "r") as f:
            config = json.load(f)
        return config

    def __configure_log_file(self, dataset_key: str) -> None:

        log_path = self.ic_config.get("log_directory")

        os.makedirs(log_path, exist_ok=True)

        start_time = datetime.now().isoformat()

        return f"{log_path}{dataset_key}_{start_time}.log"

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

    def get_time_chunk_map(self) -> np.ndarray:
        timestamps = self.get_repo_timestamps()

        session = self.repo.readonly_session("main")
        chunks = session.store.list()
        time_chunks = [chunk for chunk in chunks if "time/c/" in chunk]
        time_indexes = [int(tc.replace("time/c/", "")) for tc in time_chunks]

        return {int(ts): i for ts, i in zip(timestamps, time_indexes)}

    def create_branch(self, branch_id: str) -> None:
        branches = list(self.repo.list_branches())
        if branch_id in branches:
            return
        snap_id = self.repo.lookup_branch("main")
        self.repo.create_branch(branch_id, snapshot_id=snap_id)

    def delete_branch(self, branch_id: str) -> None:
        self.repo.delete_branch(branch_id)

    def get_dataset_nc_files(self) -> np.array:
        """
        Gets all netcdf files in the datastore using dataset data_path_template.
        """

        template_keys = self.dataset_config["template_keys"]

        nc_files = glob.glob(
            self.dataset_config["data_path_template"].format(
                **{k: "*" for k in template_keys}
            )
        )

        filter_keys = self.dataset_config.get("path_filters")
        if filter_keys:
            nc_files = [f for f in nc_files if all([k not in f for k in filter_keys])]

        return nc_files

    def get_timestamp_index(self, timestamp: int, repo_timestamps: list) -> int | None:

        time_chunk_index = None

        idx = np.argwhere(repo_timestamps == timestamp).flatten()
        if len(idx) > 0:
            time_chunk_index = int(idx[0])

        return time_chunk_index

    def nc_file_path_fcst_info(self, nc_files: list) -> pd.DataFrame:
        """
        Parses netcdf file names for forecast metadata using data_path_template.

        Args:
            nc_files: a list of NetCDF files to process.

        Returns:
            pd.DataFrame: a datafame of NetCDF files and metadata parsed from the
                    dataset's path template keys.
        """

        path_template = self.dataset_config["data_path_template"]
        # Remove file extension and wildcards
        path_template = path_template.rsplit(".", 1)[0].replace("*", "")

        template_keys = self.dataset_config["template_keys"]
        variable_map = self.dataset_config.get("variable_map")

        nc_pts = np.array([f.split("/") for f in nc_files])
        template_pts = np.array(path_template.split("/"))

        field_keys = {}
        for key in template_keys:
            for key_idx, t_pt in enumerate(template_pts):
                if key in t_pt:
                    field_keys[key] = key_idx
                    key = "{" + key + "}"
                    if len(template_pts[key_idx]) > len(key):
                        bounds = t_pt.split(key)
                        for nc_idx, nc_pt in enumerate(nc_pts[:, key_idx]):
                            i_0 = nc_pt.find(bounds[0]) + len(bounds[0])
                            nc_pt = nc_pt[i_0:]
                            i_1 = nc_pt.find(bounds[1])
                            nc_pts[nc_idx, key_idx] = nc_pt[:i_1]

        nc_df = pd.DataFrame()
        for key, key_idx in field_keys.items():
            nc_df[key] = nc_pts[:, key_idx]
        if variable_map and "variable" in template_keys:
            nc_df["variable"] = nc_df["variable"].apply(lambda v: variable_map[v])
        nc_df["file"] = nc_files

        return nc_df

    def get_nc_file_info(
        self,
        nc_files: list | None = [],
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """
        Generate an array of forcast metadata and file paths for a list of NetCDF files
        or for each file in the datastore. Can be filtered to a specific period with
        optional start_date and end_date args.

        Args:
            nc_files: a list of NetCDF files to process. This will process all files in
                    the datastore if not provided.
            start_date: the date of the initial forecast to be added to the repo
            end_date: the date of the last forecast to be added to the repo

        Returns:
            list: a list of inputs that can be passed to write_timestamp formatted
                for multiprocessing
        """

        if len(nc_files) == 0:
            nc_files = self.get_dataset_nc_files()

        nc_info = self.nc_file_path_fcst_info(nc_files)

        template_keys = self.dataset_config["template_keys"]

        if self.dataset_config.get("latest_only"):
            for k in template_keys:
                if "variable" in k:
                    continue
                nc_info = nc_info.loc[nc_info[k] == nc_info[k].max()]
        elif "forecast_date" in template_keys:
            nc_info["datetime"] = pd.to_datetime(
                nc_info["forecast_date"], format="%Y%m%d"
            )

            # filter forecast data by given dates
            if start_date:
                start_date = datetime.strptime(start_date, "%Y%m%d")
                nc_info = nc_info[nc_info["datetime"] >= start_date]

            if end_date:
                end_date = datetime.strptime(end_date, "%Y%m%d")
                nc_info = nc_info[nc_info["datetime"] <= end_date]

            # calculate timestamps
            if "forecast_run" in template_keys:
                nc_info["datetime"] = nc_info["datetime"] + pd.to_timedelta(
                    nc_info["forecast_run"].astype(int).values, unit="hours"
                )

            if "forecast_time" in template_keys:
                nc_info["datetime"] = nc_info["datetime"] + pd.to_timedelta(
                    nc_info["forecast_time"].astype(int).values, unit="hours"
                )

            if self.dataset_config.get("timestamp_offset"):
                nc_info["datetime"] = nc_info["datetime"] + pd.to_timedelta(
                    self.dataset_config.get("timestamp_offset"), unit="hours"
                )

            nc_info["timestamp"] = cftime.date2num(
                nc_info["datetime"].values.astype(datetime),
                self.dataset_config["time_dim_units"],
            )

            nc_info = nc_info.sort_values(
                by=[
                    "timestamp",
                    *template_keys,
                ]
            ).reset_index(drop=True)

            # only keep most recent timestamp for each variable
            filter_keys = [k for k in ["timestamp", "variable"] if k in nc_info.columns]
            if len(filter_keys) > 0:
                nc_info.drop_duplicates(subset=filter_keys, keep="last", inplace=True)

            # ensure each timestamp includes all variables
            ts_file_counts = nc_info.groupby(["timestamp"])["file"].transform("size")
            if "variable_map" in template_keys:
                nc_info = nc_info[
                    (ts_file_counts == len(self.dataset_config["variable_map"]))
                ]
            else:
                nc_info = nc_info[(ts_file_counts == ts_file_counts.max())]

        return nc_info

    def generate_commit_msg(
        self, nc_file_info: pd.DataFrame, status: str = "success"
    ) -> str:
        """
        Creates a commit message based on nc_file_info's columns.

        Args:
            nc_file_info: A DataFrame containing meta data for the NC files being
                            written.
            status: Whether the commit was successful, raised on error, or the files
                    were skipped. Must be "success", "error", or "skip".

        Returns:
            The compiled commit message string.
        """
        msg_info = []
        for col in nc_file_info:
            if col in ["forecast_date", "forecast_run", "forecast_time"]:
                msg_info.append(nc_file_info[col].iloc[0])

        if status == "success":
            commit_msg = "Wrote data from " + " ".join(msg_info) + " forecast."
        elif status == "error":
            commit_msg = (
                "Could not write data from " + " ".join(msg_info) + " forecast."
            )
        elif status == "skip":
            commit_msg = "Skipped data from " + " ".join(msg_info) + " forecast."

        return commit_msg

    def initialize_repo_data(self, nc_file_info: pd.DataFrame) -> pd.DataFrame:
        timestamp = nc_file_info.timestamp.min()
        ts_data = nc_file_info.loc[nc_file_info["timestamp"] == timestamp]
        parser_type = self.dataset_config.get("parser")
        drop_vars = self.dataset_config.get("drop_vars")

        match parser_type:
            case "nc3":
                parser = NetCDF3Parser()
            case _:
                parser = HDFParser()

        file_urls = [f"file://{file}" for file in ts_data["file"].values]
        store = LocalStore(prefix="/data/")
        registry = ObjectStoreRegistry({file_url: store for file_url in file_urls})

        session = self.repo.writable_session(branch="main")
        with open_virtual_mfdataset(
            urls=file_urls,
            parser=parser,
            registry=registry,
            drop_variables=drop_vars,
            compat="override",
        ) as vds:
            vds.virtualize.to_icechunk(session.store)
        session.commit(self.generate_commit_msg(ts_data))

        return nc_file_info.drop(ts_data.index)
