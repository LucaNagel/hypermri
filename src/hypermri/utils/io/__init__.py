"""
Input/Output utilities for hypermri.

This module contains I/O functions extracted from the monolithic utils_general.py
as part of the Phase 2 refactoring to create focused utility modules.
"""

from .converters import (
    export_to_csv,
    export_to_dicom,
    export_to_matlab,
    export_xarray_to_netcdf,
    load_from_matlab,
    load_xarray_from_netcdf,
    save_dict_to_excel,
    save_markdown_as_word,
)
from .dicom import DicomSeriesInfo, discover_dicom_series, load_dicom_series
from .file_utils import (
    backup_file,
    cleanup_old_files,
    create_safe_filename,
    ensure_directory,
    find_files_by_pattern,
    get_directory_size,
    get_file_info,
    sync_directories,
)
from .loaders import load_pc_sam_temp_file

__all__ = [
    # Loaders
    "load_pc_sam_temp_file",
    # Converters
    "export_to_csv",
    "export_to_matlab",
    "export_to_dicom",
    "load_from_matlab",
    "export_xarray_to_netcdf",
    "load_xarray_from_netcdf",
    "save_dict_to_excel",
    "save_markdown_as_word",
    "DicomSeriesInfo",
    "discover_dicom_series",
    "load_dicom_series",
    # File utilities
    "ensure_directory",
    "get_file_info",
    "find_files_by_pattern",
    "backup_file",
    "cleanup_old_files",
    "get_directory_size",
    "create_safe_filename",
    "sync_directories",
]
