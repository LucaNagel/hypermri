"""Registry and loader for approved HyperMRI example datasets."""

from .fetch import (
    DatasetError,
    extract_archive_safely,
    fetch_example_data,
    get_data_home,
    get_dataset_info,
    is_dataset_cached,
    list_datasets,
    load_registry,
)

__all__ = [
    "DatasetError",
    "extract_archive_safely",
    "fetch_example_data",
    "get_data_home",
    "get_dataset_info",
    "is_dataset_cached",
    "list_datasets",
    "load_registry",
]
