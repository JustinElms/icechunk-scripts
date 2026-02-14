import logging

import numpy as np
from obstore.store import LocalStore
from virtualizarr import open_virtual_mfdataset
from virtualizarr.parsers import HDFParser, NetCDF3Parser
from virtualizarr.registry import ObjectStoreRegistry

from icechunk import (
    ConflictError,
    Repository,
    VirtualChunkSpec,
)


def write_to_repo(
    repo: Repository,
    commit_msg: str,
    nc_files: list = [],
    drop_vars: list = [],
    time_chunk_index: int | None = None,
    append_dim: str = "time",
    parser_type: str = "hdf5",
    latest_only=False,
) -> None:
    """
    Initializes repository as virtual dataset with data in nc_files.

    args:
        session:
        nc_files: A Pandas Dataframe containing NetCDF files and metadata.
        append_dim: Dimension to append new data along. Set to "time" by default.
                    Setting to None will initialze or overwrite repo data.
        time_chunk_index: int | None = None,
        append_dim:
        parser_type:
        latest_only:

    returns:
        session:
    """

    logger = logging.getLogger("ic_interface")

    match parser_type:
        case "nc3":
            parser = NetCDF3Parser()
        case _:
            parser = HDFParser()

    coordinate_variables = [
        "nav_lat_u",
        "nav_lon_u",
        "nav_lat_v",
        "nav_lon_v",
        "nav_lat",
        "nav_lon",
        "latitude_u",
        "longitude_u",
        "latitude_v",
        "longitude_v",
        "latitude",
        "longitude",
        "lat",
        "lon",
    ]
    file_urls = [f"file://{file}" for file in nc_files]
    store = LocalStore(prefix="/data/")
    registry = ObjectStoreRegistry({file_url: store for file_url in file_urls})

    with open_virtual_mfdataset(
        urls=file_urls,
        parser=parser,
        registry=registry,
        drop_variables=drop_vars,
        compat="override",
    ) as vds:

        # check if any coordinates are erroniously included in data variables
        coord_data_vars = [v for v in vds.data_vars if v in coordinate_variables]
        for cdv in coord_data_vars:
            vds = vds.set_coords(cdv)

        chunks = []
        if time_chunk_index is not None and not np.isnan(time_chunk_index):
            variables = vds.data_vars
            if latest_only:
                variables.append("time")
            for variable in variables:
                chunks = []
                for manifest_idx, spec in (
                    vds[variable].data.manifest.dict().items()
                ):
                    index = list(map(int, manifest_idx.split(".")))
                    index[0] = int(time_chunk_index)
                    chunks.append(VirtualChunkSpec(index, *spec.values()))

        while True:
            try:
                session = repo.writable_session("main")
                if len(chunks) > 0:
                    session.store.set_virtual_refs(variable, chunks)
                elif append_dim:
                    vds.virtualize.to_icechunk(session.store, append_dim=append_dim)
                else:
                    vds.virtualize.to_icechunk(session.store)
                session.commit(commit_msg)
                logger.info(commit_msg)
                break
            except ConflictError:
                pass
