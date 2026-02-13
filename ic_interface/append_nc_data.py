import argparse
import logging
import multiprocessing as mp
import os
import sys
from concurrent.futures import as_completed, ProcessPoolExecutor
from logging.handlers import QueueHandler, QueueListener

from ic_interface import IcechunkInterface
from write_to_repo import write_to_repo


def init_worker(log_queue: mp.Queue, log_level: int):
    """Initialize a worker process with a QueueHandler."""
    logger = logging.getLogger("ic_interface")
    logger.setLevel(log_level)
    # Clear any existing handlers to prevent duplicate logs
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    handler = QueueHandler(log_queue)
    logger.addHandler(handler)


if __name__ == "__main__":
    mp.set_start_method("forkserver")
    max_procs = int(os.cpu_count() / 2)

    parser = argparse.ArgumentParser()
    parser.add_argument("key", help="The dataset key.", type=str)
    parser.add_argument(
        "-s",
        "--skip_existing",
        action="store_true",
        help="Skip files that match timestamps in the dataset repo. (optional)",
    )
    parser.add_argument(
        "--start_date",
        help="The earliest timestamp date to include in YYYYMMDD format. (optional)",
        default=None,
    )
    parser.add_argument(
        "--end_date",
        help="The latest timestamp date to include in YYYYMMDD format. (optional)",
        default=None,
    )
    parser.add_argument(
        "-nc",
        "--nc_files",
        help=(
            "The path to the NetCDF or other format data files to append. If not"
            "provided this script will  add all files in the datastore to the "
            "repository. (optional)"
        ),
        nargs="*",
        default=[],
        type=str,
    )

    args = parser.parse_args()
    ic_interface = IcechunkInterface(args.key)

    log_queue = mp.Queue()

    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s [%(processName)s] %(levelname)s: %(message)s"
    )
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(ic_interface.log_file)

    listener = QueueListener(log_queue, console_handler, file_handler)
    listener.start()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[console_handler, file_handler],
    )
    root_logger = logging.getLogger("ic_interface")
    root_logger.setLevel(logging.INFO)

    ic_interface = IcechunkInterface(args.key)
    initialized = ic_interface.initialized

    nc_file_info = ic_interface.get_nc_file_info(
        args.nc_files, args.start_date, args.end_date
    )

    if not ic_interface.initialized:
        nc_file_info = ic_interface.initialize_repo_data(nc_file_info)

    if len(nc_file_info) == 0:
        sys.exit()

    if ic_interface.dataset_config.get("latest_only"):
        session = ic_interface.repo.writable_session(branch="main")
        time_chunk_index = None
        if ic_interface.initialized:
            time_chunk_index = 0
        session = ic_interface.write_repo_data(
            session=session,
            nc_files=nc_file_info["file"].values,
            drop_vars=ic_interface.dataset_config.get("drop_vars"),
            time_chunk_index=time_chunk_index,
            append_dim=None,
            parser_type=parser,
            latest_only=True,
        )
        session.commit(ic_interface.generate_commit_msg(nc_file_info))

    elif "timestamp" in nc_file_info.columns:

        timestamps = sorted(nc_file_info.timestamp.values)

        with ProcessPoolExecutor(
            initializer=init_worker, initargs=(log_queue, logging.INFO)
        ) as executor:
            futures = []
            for timestamp in timestamps:
                ts_data = nc_file_info.loc[nc_file_info.timestamp == timestamp]
                commit_msg = ic_interface.generate_commit_msg(ts_data)
                futures.append(
                    executor.submit(
                        write_to_repo,
                        repo=ic_interface.repo,
                        commit_msg=commit_msg,
                        nc_files=ts_data.file.values,
                        drop_vars=[],
                        time_chunk_index=ts_data.time_chunk_index.values[0],
                        append_dim="time",
                        parser_type="hdf5",
                        latest_only=False,
                    )
                )
            [f.result() for f in futures]

    listener.stop()
    log_queue.close()
    log_queue.join_thread()
