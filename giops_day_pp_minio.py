# An example of using multiprocessing to write to an Icechunk dataset
import multiprocessing as mp
import glob
import time
import itertools
import os
import shutil
import tempfile
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, process

import numpy as np
import pandas as pd
import xarray as xr
from icechunk.xarray import to_icechunk
from icechunk import Repository, Session, local_filesystem_storage, s3_storage

N_PROCS = 4


def write_timestamp(args) -> None:
    timestamp, model_run, variable, file, session = args
    print(timestamp, model_run, variable)
    with xr.open_dataset(file, decode_times=False).drop_vars(
        ["latitude_longitude"]
    ) as ds:
        to_icechunk(ds, session, append_dim="time")
        print(f"wrote {(timestamp, model_run, variable)}")
    return session


if __name__ == "__main__":
    mp.set_start_method("forkserver")

    variables = ["vomecrty", "votemper", "vosaline", "vozocrtx"]
    model_runs = ["00"]

    model_dir = "/data/hpfx.collab.science.gc.ca/"
    model_runs = "/data/hpfx.collab.science.gc.ca/{timestamp}/WXO-DD/model_giops/netcdf/lat_lon/3d/00/000/*{timestamp}*.nc"

    timestamps = sorted(glob.glob(f"{model_dir}/*"))
    timestamps = np.array([ts.replace(model_dir, "") for ts in timestamps][:-1]).astype(
        int
    )
    timestamps = timestamps[-10:]

    nc_data = list(itertools.product(timestamps, ["00"], variables))

    for idx, nd in enumerate(nc_data):
        nc_files = glob.glob(
            f"/data/hpfx.collab.science.gc.ca/{nd[0]}/WXO-DD/model_giops/netcdf/lat_lon/3d/{nd[1]}/*/*{nd[2]}*.nc"
        )
        nc_data[idx] = [*nd, nc_files[0]]

    nc_df = pd.DataFrame(
        nc_data, columns=["timestamp", "model_run", "variable", "file"]
    )

    storage_config = s3_storage(
        bucket="icechunk",
        prefix="giops_day",
        access_key_id="admin",
        secret_access_key="password",
        endpoint_url="http://142.130.249.35:9000",
        allow_http=True,
        force_path_style=True,
    )

    if not Repository.exists(storage_config):
        repo = Repository.create(storage_config)
        for var in variables:
            idx = nc_df.loc[nc_df["variable"] == var].index[0]
            nc_info = nc_df.loc[idx].values.tolist()
            session = repo.writable_session("main")
            with xr.open_dataset(nc_info[-1], decode_times=False).drop_vars(
                ["latitude_longitude"]
            ) as ds:
                to_icechunk(ds, session, mode="a")
            session.commit(
                f"Wrote variable {nc_info[2]} from model run {nc_info[1]} of timestamp {nc_info[0]}"
            )
            nc_df.drop(idx, inplace=True)
    else:
        repo = Repository.open(storage_config)

    nc_chunks = np.array_split(nc_df.values, np.ceil(len(nc_df) / N_PROCS))
    t0 = time.time()
    for nc_chunk in nc_chunks:
        print(f"Processing files {', '.join(nc_chunk[:, -1].astype(str))}")

        session = repo.writable_session("main")
        with ProcessPoolExecutor(max_workers=N_PROCS) as executor:
            # submit the writes
            fork = session.fork()
            nc_chunk = np.append(
                nc_chunk,
                np.repeat(fork, len(nc_chunk)).reshape((len(nc_chunk), 1)),
                axis=1,
            )
            remote_sessions = executor.map(write_timestamp, nc_chunk)

        session.merge(*remote_sessions)
        session.commit(f"Wrote files {', '.join(nc_chunk[:,-1].astype(str))}")

    t1 = time.time()
    print(f"Completed in {t1 - t0} seconds")
