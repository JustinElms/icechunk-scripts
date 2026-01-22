import argparse
import gc
import os
import pickle
import psutil
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import ic_manager

N_PROC = 4


def get_system_memory_info():
    svmem = psutil.virtual_memory()
    return {
        "total": svmem.total / (1024 * 1024),
        "available": svmem.available / (1024 * 1024),
        "percent": svmem.percent,
    }


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    n_procs = os.cpu_count()

    dataset_key = "giops_day"

    manager = ic_manager.ic_manager(dataset_key)

    pkl_name = f"{dataset_key}_init_data.pkl"
    if os.path.exists(pkl_name):
        with open(pkl_name, "rb") as f:
            nc_file_info = pickle.load(f)
    else:
        nc_file_info = manager.init_nc_file_data()

        with open(pkl_name, "wb") as f:
            pickle.dump(nc_file_info, f)

    timestamps = nc_file_info["timestamp"].unique()

    initialized = manager.init_repo_data(
        nc_file_info.loc[nc_file_info["timestamp"] == timestamps[0], "file"], timestamps[0]
    )

    drop_vars = manager.dataset_config.get("drop_vars")

    repo_timestamps = manager.get_repo_timestamps()

    for timestamp in timestamps:
        if timestamp in repo_timestamps:
            continue
        except_count = 0
        ts_data = nc_file_info.loc[nc_file_info["timestamp"] == timestamp, "file"]
        append_dim = "time"
        if timestamp == timestamps[0] and initialized:
            append_dim = None
        while True:
            print(get_system_memory_info())
            try:
                session = manager.repo.writable_session("main")
                with ProcessPoolExecutor() as executor:
                    fork = session.fork()
                    futures = [
                        executor.submit(
                            manager.append_timestamp,
                            nc_file=f,
                            drop_vars=drop_vars,
                            append_dim=append_dim,
                            session=fork,
                        )
                        for f in ts_data.values
                    ]

                    remote_sessions = [f.result() for f in futures]

                session.merge(*remote_sessions)
                session.commit(f"Wrote {len(ts_data)} files from timestamp {timestamp}")
                nc_file_info.drop(ts_data.index, inplace=True)
                break
            except Exception as e:
                print(
                    f"Could not write {len(ts_data)} files from timestamp {timestamp}, retrying."
                )
                print(e)
                print(gc.collect())
                pass
        print(f"Wrote files: \n{',\n'.join(ts_data.values)}")
