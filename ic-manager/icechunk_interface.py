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
    ConflictError,
    ForkSession,
    VirtualChunkContainer,
    local_filesystem_store,
)
from obstore.store import LocalStore
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser
from virtualizarr.registry import ObjectStoreRegistry


class IcechunkInterface:

    def __init__(self, dataset_key: str) -> None:
        self.configs_dir = "/home/ubuntu/icechunk/ic-configs"
        self.log_file = "ic_log.txt"

        self.dataset_key = dataset_key
        self.dataset_config = self.__read_config()
        self.nc_files = []

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
            virtual_config = VirtualChunkContainer(
                url_prefix="file:///data/", store=local_filesystem_store(path="/data/")
            )
            repo_config.set_virtual_chunk_container(virtual_config)
            self.repo = Repository.create(storage_config, config=repo_config)
            self.repo.save_config()
        else:
            self.repo = Repository.open(
                storage_config, authorize_virtual_chunk_access={"file:///data/": None}
            )

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

    def get_forecast_dates(self, start_date: str, end_date: str | None = None) -> list:
        date = datetime.strptime(start_date, "%Y%m%d")
        if not end_date:
            end = datetime.now()
        else:
            end = datetime.strptime(end_date, "%Y%m%d")

        forecast_dates = []
        while date <= end:
            forecast_dates.append(date)
            date += timedelta(days=1)

        return [datetime.strftime(fd, "%Y%m%d") for fd in forecast_dates]

    def calculate_chunk_size(self, variable: xr.DataArray, target_size: int) -> tuple:
        """
        Calculates a chunk size smaller than target_size for the given variable.

        args:
            variable - an xarray DataArray for the selected variable
            target_size - the maximum chunk size in Mb

        returns:

        """
        chunks = [variable[d].size for d in variable.dims]
        size = np.prod(chunks) * np.dtype(variable.dtype).itemsize / 1e6

        dim_idx = len(chunks) - 1
        while size > target_size:
            new_chunks = chunks
            new_chunks[dim_idx] = int(new_chunks[dim_idx] / 2)
            dim_idx -= 1
            if dim_idx < 1:
                dim_idx = len(chunks) - 1

            size = np.prod(new_chunks) * np.dtype(variable.dtype).itemsize / 1e6

            if size < 1:
                break
            else:
                chunks = new_chunks

        return tuple(chunks)

    def create_branch(self, branch_id: str) -> None:
        branches = list(self.repo.list_branches())
        if branch_id in branches:
            return
        snap_id = self.repo.lookup_branch("main")
        self.repo.create_branch(branch_id, snapshot_id=snap_id)

    def delete_branch(self, branch_id: str) -> None:
        self.repo.delete_branch(branch_id)

    def init_repo_zarr(self, nc_files: list | pd.Series, timestamp: int) -> bool:
        """
        Initializes repository with data in nc_files.

        args:
            nc_files - A list or pandas series of NetCDF files
            timestamp - The initial timestamp of the new repo

        returns:
            False if repo has already been initialized, True if not
        """

        if len(list(self.repo.ancestry(branch="main"))) > 1:
            print("Repository already initialized")
            return False

        encoding = {}
        with xr.open_mfdataset(nc_files).drop_vars(
            self.dataset_config["drop_vars"]
        ) as ds:
            for var in ds.data_vars:
                chunks = self.calculate_chunk_size(ds[var], 50)
                encoding[var] = {"chunks": chunks}
                ds[var] = ds[var].chunk(chunks)

            session = self.repo.writable_session("main")
            ds.to_zarr(
                session.store,
                compute=False,
                encoding=encoding,
                mode="w",
                consolidated=False,
            )

        session.commit(f"Initialized repo with data from timestamps {timestamp}")
        return True

    def init_repo_virtual(self, nc_files: list | pd.Series, timestamp: int) -> bool:
        """
        Initializes repository as virtual dataset with data in nc_files.

        args:
            nc_files - A list or pandas series of NetCDF files
            timestamp - The initial timestamp of the new repo

        returns:
            False if repo has already been initialized, True if not
        """

        if len(list(self.repo.ancestry(branch="main"))) > 1:
            print("Repository already initialized")
            return False

        file_urls = [f"file://{nc_file}" for nc_file in nc_files]
        store = LocalStore(prefix="/data/")
        registry = ObjectStoreRegistry({file_url: store for file_url in file_urls})

        with open_virtual_mfdataset(
            urls=file_urls,
            parser=HDFParser(),
            registry=registry,
            drop_variables=self.dataset_config["drop_vars"],
        ) as vds:
            session = self.repo.writable_session("main")
            vds.virtualize.to_icechunk(session.store)

        session.commit(f"Initialized repo with data from timestamps {timestamp}")
        return True

    def get_datastore_nc_files(self) -> np.array:
        """
        Gets all netcdf files in the datastore using dataset data_path_template.
        """

        nc_files = glob.glob(
            self.dataset_config["data_path_template"].format(
                forecast_date="*",
                forecast_run="*",
                forecast_time="*",
                forecast_variable="*",
            )
        )

        return self.nc_file_path_fcst_info(nc_files)

    def nc_file_path_fcst_info(self, nc_files: list) -> np.array:
        """
        Parses netcdf file names for forecast metadata using data_path_template.
        """
        pts = [f.split("/") for f in nc_files]
        template_pts = self.dataset_config["data_path_template"].split("/")
        template_fields = [
            "forecast_date",
            "forecast_run",
            "forecast_time",
            "forecast_variable",
        ]
        field_keys = {}
        for f in template_fields:
            for i, p in enumerate(template_pts):
                if f in p:
                    field_keys[f] = i

        fcst_info = []
        for i, p in enumerate(pts):
            v = [
                v
                for v in self.dataset_config["variables"]
                if v in p[field_keys["forecast_variable"]].lower()
            ][0]
            fcst_info.append(
                [*[p[f] for f in list(field_keys.values())[:-1]], v, nc_files[i]]
            )

        return np.array(fcst_info)

    def init_nc_file_data(
        self, start_date: str | None = None, end_date: str | None = None
    ) -> None:
        """
        Generate an array of forcast metadata and file paths for each file in the
        datastore. Can be filtered to a specific period with optional start_date
        and end_date args.

        Args:
            start_date: the date of the initial forecast to be added to the repo
            end_date: the date of the last forecast to be added to the repo

        Returns:
            list: a list of inputs that can be passed to write_timestamp formatted
                for multiprocessing
        """

        file_info = self.get_datastore_nc_files()

        input_df = pd.DataFrame(
            file_info,
            columns=[
                "forecast_date",
                "forecast_run",
                "forecast_time",
                "variable",
                "file",
            ],
        )

        input_df["datetime"] = pd.to_datetime(
            input_df["forecast_date"], format="%Y%m%d"
        )

        # filter forecast data by given dates
        if start_date:
            start_date = datetime.strptime(start_date, "%Y%m%d")
            input_df = input_df[input_df["datetime"] >= start_date]

        if end_date:
            end_date = datetime.strptime(end_date, "%Y%m%d")
            input_df = input_df[input_df["datetime"] <= end_date]

        # calculate timestamps
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
        input_df.drop_duplicates(
            subset=["timestamp", "variable"], keep="last", inplace=True
        )

        ts_file_counts = input_df.groupby(["timestamp"])["file"].transform("size")
        input_df = input_df[(ts_file_counts == len(self.dataset_config["variables"]))]

        return input_df

    @staticmethod
    def append_timestamp(
        *, nc_file: str, drop_vars: list, append_dim: str = None, session: ForkSession
    ) -> None:
        print(f"Writing {nc_file}")
        ds = xr.open_mfdataset(nc_file, decode_times=False).drop_vars(drop_vars)
        if append_dim:
            ds.to_zarr(
                session.store,
                mode="a",
                append_dim=append_dim,
                consolidated=False,
                align_chunks=True,
            )
        else:
            ds.to_zarr(session.store, mode="a", consolidated=False, align_chunks=True)
        return session

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

    @staticmethod
    def write_timestamp_uncoop(args) -> None:
        drop_vars, mode, append_dim, region, file, repo = args
        print(f"Processing file {file}")
        while True:
            try:
                session = repo.writable_session("main")
                with xr.open_dataset(file, decode_times=False).drop_vars(
                    drop_vars
                ) as ds:
                    if mode == "r+":
                        to_icechunk(ds, session, mode=mode, region=region)
                    elif append_dim:
                        to_icechunk(ds, session, mode=mode, append_dim=append_dim)
                    else:
                        to_icechunk(ds, session, mode=mode)
                session.commit(f"Wrote file {file}")
                print(f"Wrote file {file}")
                break
            except ConflictError as e:
                print(e)
                print(f"Conflict for {file}, retrying")
                pass
