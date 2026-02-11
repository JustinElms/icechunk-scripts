import json

import zarr
from icechunk import (
    Repository,
    s3_storage,
)


def __get_storage(self, dataset_key, s3_config) -> s3_storage:
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


if __name__ == "__main__":
    """
    Adds the missing valid_min and valid_max attributes to POPS forecast variables.
    Can only be run after the repository has been initialized.
    """

    with open("ic_config.json", "r") as f:
        ic_config = json.load(f)

    s3_config = ic_config.get("s3_config")

    pops_dataset_keys = [
        "canso100",
        "canso500",
        "fundy500",
        "kit100",
        "kit500",
        "sf30",
        "sj100",
        "sss150",
        "stle200",
        "stle500",
        "vh20",
    ]

    valid_range_attrs = {
        "uos": {"valid_min": -20.0, "valid_max": 20.0},
        "vos": {"valid_min": -20.0, "valid_max": 20.0},
        "sos": {"valid_min": 0.0, "valid_max": 45.0},
        "tos": {"valid_min": 173.0, "valid_max": 373.0},
    }

    for dataset_key in pops_dataset_keys:
        storage_config = s3_storage(
            bucket=s3_config.get("bucket"),
            prefix=dataset_key,
            region="us-east-1",
            access_key_id=s3_config.get("user"),
            secret_access_key=s3_config.get("password"),
            endpoint_url=s3_config.get("url"),
            allow_http=True,
            force_path_style=True,
        )

        repo = Repository.open(
            storage_config, authorize_virtual_chunk_access={"file:///data/": None}
        )

        session = repo.writable_session("main")

        group = zarr.open_group(session.store)

        for key, array in group.arrays():
            if key in valid_range_attrs.keys():
                next_attrs = valid_range_attrs[key]
                for k, v in next_attrs.items():
                    array.attrs[k] = v

        session.commit(message=f"Added missing attributes to {dataset_key}.")
