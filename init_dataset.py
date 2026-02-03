import argparse

from ic_interface.icechunk_interface import IcechunkInterface


def init_ic_repo(dataset_key: str) -> None:
    ic_interface = IcechunkInterface(dataset_key)

    nc_file_info = ic_interface.init_nc_file_data()
    drop_vars = ic_interface.dataset_config.get("drop_vars")

    if not ic_interface.initialized:
        timestamp = nc_file_info.timestamp.min()
        ts_data = nc_file_info.loc[nc_file_info["timestamp"] == timestamp]

        session = ic_interface.repo.writable_session(branch="main")
        session = ic_interface.write_repo_data(
            session=session,
            nc_files=ts_data["file"].values,
            drop_vars=drop_vars,
            append_dim=None,
        )
        session.commit(f"Initialized with timestamp {timestamp} data.")

        nc_file_info.drop(ts_data.index, inplace=True)

    timestamps = sorted(nc_file_info["timestamp"].unique())
    for timestamp in timestamps:
        ts_data = nc_file_info.loc[nc_file_info["timestamp"] == timestamp]

        session = ic_interface.repo.writable_session(branch="main")

        ic_interface.write_repo_data(
            session=session,
            nc_files=ts_data["file"].values,
            drop_vars=drop_vars,
            time_chunk_index=ts_data["time_chunk_index"].values[0],
        )

        commit_msg = "Wrote timestamp %s from %s %s %s forecast." % (
            ts_data.timestamp.iloc[0],
            ts_data.forecast_date.iloc[0],
            ts_data.forecast_run.iloc[0],
            ts_data.forecast_time.iloc[0],
        )

        session.commit(commit_msg)
        print(commit_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-k", "--key", help="The key of the dataset to initialize.", type=str
    )
    args = parser.parse_args()

    init_ic_repo(args.key)
