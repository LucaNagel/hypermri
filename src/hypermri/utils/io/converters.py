"""
File format conversion utilities.

Contains functions for exporting data to various formats like Word, Excel, CSV, MATLAB, and NetCDF.
"""

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import scipy.io as sio
import xarray as xr

from ..system.logging import init_default_logger

logger = init_default_logger(__name__)


def save_markdown_as_word(
    input_md: str | None = None,
    file_path: str | None = None,
    file_name: str | None = None,
    print_file_path: bool = True,
) -> None:
    """
    Convert a Markdown string to a Word (.docx) file using Pandoc via subprocess.

    Parameters
    ----------
    input_md : str or None
        The protocol content in Markdown format.
    file_path : str or None
        The directory path where the resulting .docx file should be saved.
    file_name : str or None
        The name of the output Word document.
    print_file_path : bool, optional
        If True, prints the final file path upon successful creation.

    Notes
    -----
    Pandoc must be installed on your system.
    """
    # Write the Markdown text to a temporary .md file
    temp_md = "temp_protocol.md"
    with open(temp_md, "w", encoding="utf-8") as f:
        f.write(input_md if input_md else "")

    # Construct the full output path
    if file_path is None:
        file_path = os.getcwd()
    output_docx_path = os.path.join(file_path, file_name) if file_name else None

    try:
        # Invoke Pandoc to convert Markdown to .docx
        subprocess.run(["pandoc", temp_md, "-o", output_docx_path], check=True)
        if print_file_path:
            logger.info(f"Saved Word file to: {output_docx_path}")
    except FileNotFoundError:
        logger.error("Pandoc not found. Please install pandoc to use this feature.")
    except Exception as e:
        logger.error(f"Error converting markdown to word: {e}")
    finally:
        # Remove the temporary file
        if os.path.exists(temp_md):
            os.remove(temp_md)


def save_dict_to_excel(
    data_dict, savepath, filename, sheet_name=None, header=None, index=False
):
    """
    Saves a dictionary to an Excel file.

    Parameters
    ----------
    data_dict : dict
        Dictionary to save.
    savepath : str
        Directory path.
    filename : str
        Name of the Excel file.
    sheet_name : str, optional
        Name of the sheet.
    header : list, optional
        Column names.
    index : bool, default False
        Whether to write row indices.
    """
    os.makedirs(savepath, exist_ok=True)
    file_path = os.path.join(savepath, f"{filename}.xlsx")
    sheet_name = sheet_name if sheet_name else "Sheet1"

    df = pd.DataFrame(
        list(data_dict.items()), columns=header if header else ["Key", "Value"]
    )

    if os.path.exists(file_path):
        with pd.ExcelWriter(
            file_path, engine="openpyxl", mode="a", if_sheet_exists="new"
        ) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)
    else:
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="w") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=index)

    logger.info(f"Data saved to: {file_path}, sheet: {sheet_name}")


def export_to_csv(
    data: Union[pd.DataFrame, np.ndarray, dict], filepath: Union[str, Path], **kwargs
) -> bool:
    """
    Export data to CSV format.

    Parameters
    ----------
    data : DataFrame, ndarray, or dict
        Data to export.
    filepath : str or Path
        Output file path.
    **kwargs
        Additional arguments for pandas.to_csv().

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    filepath = Path(filepath)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, **kwargs)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame(data, columns=["Value"])
            elif data.ndim == 2:
                df = pd.DataFrame(data)
            else:
                logger.error("Cannot export arrays with >2 dimensions to CSV")
                return False
            df.to_csv(filepath, **kwargs)
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
            df.to_csv(filepath, **kwargs)
        else:
            logger.error(f"Unsupported data type for CSV export: {type(data)}")
            return False

        logger.info(f"Data exported to CSV: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error exporting to CSV {filepath}: {e}")
        return False


def export_to_matlab(data: Dict[str, Any], filepath: Union[str, Path]) -> bool:
    """
    Export data to MATLAB .mat format.

    Parameters
    ----------
    data : dict
        Data dictionary to export.
    filepath : str or Path
        Output file path.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    filepath = Path(filepath)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix != ".mat":
            filepath = filepath.with_suffix(".mat")

        sio.savemat(str(filepath), data)
        logger.info(f"Data exported to MATLAB format: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error exporting to MATLAB {filepath}: {e}")
        return False


def load_from_matlab(filepath: Union[str, Path]) -> Union[Dict[str, Any], None]:
    """
    Load data from MATLAB .mat format.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    dict or None
        Loaded data dictionary, or None if failed.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    try:
        data = sio.loadmat(str(filepath))
        # Remove MATLAB metadata
        cleaned_data = {k: v for k, v in data.items() if not k.startswith("__")}
        logger.info(f"Data loaded from MATLAB format: {filepath}")
        return cleaned_data

    except Exception as e:
        logger.error(f"Error loading from MATLAB {filepath}: {e}")
        return None


def export_xarray_to_netcdf(data, filepath: Union[str, Path], **kwargs) -> bool:
    """
    Export xarray DataArray/Dataset to NetCDF format.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Data to export.
    filepath : str or Path
        Output file path.
    **kwargs
        Additional arguments for to_netcdf().

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    filepath = Path(filepath)

    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        if filepath.suffix != ".nc":
            filepath = filepath.with_suffix(".nc")

        data.to_netcdf(str(filepath), **kwargs)
        logger.info(f"xarray data exported to NetCDF: {filepath}")
        return True

    except Exception as e:
        logger.error(f"Error exporting xarray to NetCDF {filepath}: {e}")
        return False


def load_xarray_from_netcdf(filepath: Union[str, Path]):
    """
    Load xarray data from NetCDF format.

    Parameters
    ----------
    filepath : str or Path
        Input file path.

    Returns
    -------
    xarray.DataArray or xarray.Dataset or None
        Loaded data, or None if failed.
    """
    import xarray as xr

    filepath = Path(filepath)

    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return None

    try:
        data = xr.open_dataset(str(filepath))
        logger.info(f"xarray data loaded from NetCDF: {filepath}")
        return data

    except Exception as e:
        logger.error(f"Error loading xarray from NetCDF {filepath}: {e}")
        return None


def export_to_dicom(
    data: Union["xr.DataArray", np.ndarray],
    output_dir: Union[str, Path],
    reference_dicoms=None,
    filename_prefix: str = "MRIm",
    series_description: str = "Exported Data",
    series_number: int = 999,
    unit: str = None,
    *,
    geometry=None,
    allow_identity_geometry: bool = False,
) -> bool:
    """
    Export image data as a derived MR DICOM series and report success.

    This is the public entry point of
    :func:`hypermri.utils.io.dicom_export.export_to_dicom`, which documents
    the data contract, the geometry sources and the written DICOM attributes
    in full.  Unlike that function, it does not raise: any error is logged
    at error level, which the default log level shows, and False is
    returned.

    Parameters
    ----------
    data : xarray.DataArray or numpy.ndarray
        Image values with the dimensions ``read_idx`` and ``phase_idx`` and
        optionally ``slice_idx``; other dimensions become frames.  A NumPy
        array is read as ``(phase_idx, read_idx[, slice_idx])`` and has no
        geometry of its own. ``slice_thickness_mm`` and ``slice_spacing_mm``
        in slice-coordinate or array attributes preserve physical thickness
        separately from sampling distance, including after slice selection.
        Used sizes must be positive finite scalars in mm. The input is not
        modified.
    output_dir : str or pathlib.Path
        Directory for the DICOM files; created when missing.
    reference_dicoms : object, optional
        Reference DICOM series (directory, series info, Dataset or sequence
        of Datasets, or a scan object with loaded DICOMs).  Supplies the
        geometry when ``data`` and ``geometry`` do not, and its patient and
        study context is copied.
    filename_prefix : str, optional
        File-name prefix; files are named ``<prefix><instance:04d>.dcm``.
    series_description : str, optional
        ``SeriesDescription`` of the new series.
    series_number : int, optional
        ``SeriesNumber`` of the new series.
    unit : str, optional
        Unit of the rescaled values, written to ``RescaleType``.
    geometry : object, optional
        Geometry source when ``data`` has no ``x``, ``y`` and ``z`` (mm,
        DICOM LPS): a DataArray or Dataset with these coordinates, or a scan
        object with Bruker ``VisuCore`` geometry and the same spatial shape.
    allow_identity_geometry : bool, optional
        If True and no geometry is found, write a centred 1 mm identity grid
        and label the series (``SeriesDescription`` suffix
        ``" [identity geometry]"``, ``ImageComments`` and a
        ``UserWarning``).  Defaults to False, which makes data without
        geometry fail.

    Returns
    -------
    bool
        True if all instances were written; False if the export failed, for
        example because no geometry was found.  Nothing is written when the
        input or the geometry is invalid.

    Warns
    -----
    UserWarning
        If the identity geometry is written.

    See Also
    --------
    hypermri.utils.io.dicom_export.export_to_dicom : Raising variant with
        the full description of the export.

    Examples
    --------
    >>> from hypermri.utils.io import export_to_dicom
    >>> export_to_dicom(t1_map, "t1_dicom", geometry=scan, unit="s")  # doctest: +SKIP
    True
    """
    from .dicom_export import export_to_dicom as export_geometry_aware_dicom

    try:
        return export_geometry_aware_dicom(
            data=data,
            output_dir=output_dir,
            reference_dicoms=reference_dicoms,
            filename_prefix=filename_prefix,
            series_description=series_description,
            series_number=series_number,
            unit=unit,
            geometry=geometry,
            allow_identity_geometry=allow_identity_geometry,
        )
    except Exception as exc:
        logger.error("DICOM export failed: %s", exc)
        return False
