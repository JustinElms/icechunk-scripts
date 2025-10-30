import argparse
import itertools
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import numpy as np

import ic_manager

N_PROCS = 4


def get_date_bounds(start_date: str, end_date: str) -> list:
    date = datetime.strptime(start_date, "%Y%m%d")
    end = datetime.strptime(end_date, "%Y%m%d")

    date_bounds = []
    while date <= end:
        d0 = date.replace(day=1)
        d1 = date + timedelta(days=31)
        d1 = d1.replace(day=1) - timedelta(days=1)

        date_bounds.append(
            [datetime.strftime(d0, "%Y%m%d"), datetime.strftime(d1, "%Y%m%d")]
        )
        date = d1 + timedelta(days=1)

    return date_bounds


if __name__ == "__main__":
    mp.set_start_method("forkserver")

    dataset_key = "giops_fc_10d_2dll"
    start_date = "20240722"
    end_date = "20250918"

    manager = ic_manager.ic_manager(dataset_key)
    # ingest one month of data at a time
    date_bounds = get_date_bounds(start_date, end_date)
    for date_bound in date_bounds:
        input_groups = manager.init_repo_data(date_bound[0], date_bound[1])

        for input_group in input_groups:
            file_names = [i[-1] for i in input_group]
            print(f"Processing files: \n{',\n'.join(file_names)}")

            session = manager.repo.writable_session("main")
            with ProcessPoolExecutor(max_workers=N_PROCS) as executor:
                fork = session.fork()
                input_group = np.append(
                    input_group,
                    np.repeat(fork, len(input_group)).reshape((len(input_group), 1)),
                    axis=1,
                )
                remote_sessions = executor.map(manager.write_timestamp, input_group)

            session.merge(*remote_sessions)
            session.commit(f"Wrote files: {', '.join(file_names)}")
            print(f"Wrote files: \n{',\n'.join(file_names)}")
