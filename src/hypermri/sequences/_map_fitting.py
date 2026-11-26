"""Fit explicitly supported scanner image series through named analysis functions.

Sequence objects own results; this module validates metadata and coordinates,
selects seq2d once and records execution outcomes. It contains no fit equations.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
import xarray as xr

from hypermri.analysis.fitting.execution import ExecutionConfig
from hypermri.analysis.fitting.models.quantitative import t2
from hypermri.analysis.quantitative import ADCFitOptions, T2FitOptions, fit_adc, fit_t2
from hypermri.analysis.quantitative.options import _options_mapping
from hypermri.loading.experiment import _handler_vendor_name
from hypermri.reconstruction.strategies.bruker.seq2d import (
    _frame_group_axes,
    _frame_group_coordinates,
    _frame_group_entries,
)

_RELEASED_MODELS = ("adc", "t2")


def _select_models(models, *, automatic=False):
    """Validate selection once for constructor, Study and scan entry points."""
    if automatic and isinstance(models, bool):
        return _RELEASED_MODELS if models else ()
    if models is None and not automatic:
        return _RELEASED_MODELS
    if not isinstance(models, list | tuple):
        example = '["adc"]'
        raise TypeError(f"Model selection requires a list or tuple, e.g. {example}.")
    selected = []
    for name in models:
        if not isinstance(name, str):
            raise TypeError("Each model name must be a string, e.g. ['adc'].")
        if name not in _RELEASED_MODELS:
            raise ValueError(
                f"Unknown map model {name!r}; available models: {_RELEASED_MODELS}."
            )
        if name not in selected:
            selected.append(name)
    return tuple(selected)


def _validate_adc_options(adc_options):
    """Reject invalid supplied settings before reading data or clearing results."""
    if adc_options is None:
        return
    if not isinstance(adc_options, ADCFitOptions):
        raise TypeError("adc_options must be ADCFitOptions or None.")
    if adc_options.dynamic_dim not in (None, "bvalue_idx"):
        raise ValueError("scan ADC fitting requires dynamic_dim='bvalue_idx'.")
    if adc_options.b_values_s_per_mm2 is not None:
        raise ValueError(
            "scan ADC fitting obtains b-values from acquisition metadata; omit "
            "b_values_s_per_mm2. Use fit_adc for explicitly supplied data axes."
        )
    threshold = adc_options.signal_threshold
    if threshold is not None and (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or threshold < 0
    ):
        raise ValueError("ADC signal_threshold must be finite and nonnegative.")
    execution = adc_options.execution
    if execution is not None:
        if not isinstance(execution, ExecutionConfig):
            raise TypeError("ADC execution must be ExecutionConfig or None.")
        if not isinstance(execution.use_multiprocessing, bool):
            raise TypeError("use_multiprocessing must be a boolean.")
        workers = execution.number_of_cpu_cores
        if (
            isinstance(workers, bool)
            or not isinstance(workers, Integral)
            or workers == 0
        ):
            raise ValueError("number_of_cpu_cores must be a nonzero integer.")


def _validate_options(adc_options, t2_options=None):
    _validate_adc_options(adc_options)
    if t2_options is not None:
        if not isinstance(t2_options, T2FitOptions):
            raise TypeError("t2_options must be T2FitOptions or None.")
        if t2_options.dynamic_dim not in (None, "echo_idx"):
            raise ValueError("scan T2 fitting requires dynamic_dim='echo_idx'.")
        if t2_options.execution is not None and not isinstance(
            t2_options.execution, ExecutionConfig
        ):
            raise TypeError("T2 execution must be ExecutionConfig or None.")
        t2._validate_options(_options_mapping(t2_options))


def _supports_model(scan, model):
    # Imports on use avoid a cycle through the shared anatomical base class.
    from .DTI import DTI
    from .DTIEPI import DTIEPI
    from .MSME import MSME

    supported = {DTI: ("adc",), DTIEPI: ("adc",), MSME: ("t2",)}
    return (
        model in supported.get(type(scan), ())
        and _handler_vendor_name(scan.handler) == "bruker"
    )


def _numeric_values(value):
    """Return finite metadata values without substituting missing indices."""
    if value is None:
        return None
    try:
        values = np.asarray(value, dtype=float).reshape(-1)
    except (ValueError, TypeError):
        return None
    return values if values.size and np.all(np.isfinite(values)) else None


def _scanner_source(scan, data):
    """Validate selected reconstruction identity and original image frame groups."""
    number = data.attrs.get("bruker_reconstruction_number")
    if isinstance(number, bool) or not isinstance(number, Integral) or number < 1:
        return None, None, "Scanner reconstruction identity is missing or invalid."
    source = data.attrs.get("bruker_visu_parameter_source")
    if source is not None and source != f"pdata/{number}/visu_pars":
        return None, None, "Scanner reconstruction and Visu parameter source disagree."
    _, visu = scan.handler.load_reconstruction_parameters(int(number))
    groups = _frame_group_entries(visu)
    if "map_idx" in data.dims or any(g[1] in {"FG_DTI", "FG_ISA"} for g in groups):
        return None, None, "Scanner parameter maps are not acquired image series."
    frame_types = np.asarray(visu.get("VisuCoreFrameType", [])).reshape(-1)
    image_types = {"MAGNITUDE_IMAGE", "REAL_IMAGE", "IMAGINARY_IMAGE", "COMPLEX_IMAGE"}
    if not frame_types.size or any(
        str(v).strip("<>") not in image_types for v in frame_types
    ):
        return (
            None,
            None,
            "VisuCoreFrameType does not establish an original image series.",
        )
    return groups, visu, ""


def _adc_eligibility(scan, data, groups):
    """Require complete acquisition b-values and one explicitly declared direction."""
    dim = "bvalue_idx"
    if dim not in data.dims or dim not in data.coords:
        return "The diffusion dimension bvalue_idx or its coordinate is missing."
    size = data.sizes[dim]
    diffusion_groups = [g for g in groups if g[1] == "FG_DIFFUSION"]
    if len(diffusion_groups) != 1 or diffusion_groups[0][0] != size:
        return "FG_DIFFUSION does not establish the image-to-b-value order."
    method = scan.handler.method
    values = _numeric_values(method.get("PVM_DwEffBval"))
    if values is None or values.size != size or np.any(values < 0):
        return "PVM_DwEffBval must contain one finite nonnegative b-value per image."
    if np.unique(values).size < 2:
        return "Only one distinct b-value is available."
    coordinate = data.coords[dim]
    if (
        coordinate.dims != (dim,)
        or coordinate.attrs.get("unit") != "s/mm2"
        or not np.array_equal(coordinate.values, values)
    ):
        return "The b-value axis is an index or differs from acquisition values/order."
    count = _numeric_values(method.get("PVM_DwNDiffExp"))
    if count is None or count.size != 1 or count[0] != size:
        return "PVM_DwNDiffExp does not establish the diffusion image count."
    directions = _numeric_values(method.get("PVM_DwNDiffDir"))
    vectors = _numeric_values(method.get("PVM_DwDir"))
    if directions is not None and (directions.size != 1 or directions[0] != 1):
        return "Multiple diffusion directions are not supported by the scalar ADC path."
    if (
        directions is None
        or vectors is None
        or vectors.size != 3
        or np.linalg.norm(vectors) == 0
    ):
        return "Diffusion directions are not unambiguously assigned by PVM_DwNDiffDir/PVM_DwDir."
    return ""


def _t2_eligibility(scan, data, groups, visu):
    dim = "echo_idx"
    echo_groups = [g for g in groups if g[1] == "FG_ECHO"]
    if (
        dim not in data.dims
        or len(echo_groups) != 1
        or echo_groups[0][0] != data.sizes[dim]
    ):
        return "FG_ECHO does not establish the echo image series/order."
    try:
        times = t2._echo_times(data, dim)
        # Validate declared timing with the reconstruction's existing dependency
        # reader, using the stored unsqueezed frame layout rather than viewer axes.
        spatial = tuple(int(n) for n in visu.get("VisuCoreSize", []) if int(n) > 1)
        shape = spatial + tuple(
            g[0] for g in groups if g[0] > 1 and g[1] != "FG_COMPLEX"
        )
        axes = _frame_group_axes(visu, shape, frame_groups=groups)
        dims = tuple(axes.get(i, f"spatial_{i}") for i in range(len(shape)))
        declared = _frame_group_coordinates(visu, shape, dims, frame_groups=groups).get(
            dim
        )
    except (TypeError, ValueError, IndexError) as exc:
        return f"Echo timing is not established: {exc}"
    if declared is not None:
        expected = np.asarray(declared[1], dtype=float)
    else:
        values = _numeric_values(scan.handler.method.get("EffectiveTE"))
        if values is None or values.size != times.size or np.any(values < 0):
            return (
                "EffectiveTE must contain one finite nonnegative echo time per image."
            )
        expected = values / 1000.0
    if not np.allclose(times, expected, rtol=0, atol=1e-15):
        return (
            "Echo times differ from the selected scanner/acquisition values or order."
        )
    return ""


def _status(state, reason="", valid=None, total=None):
    return {
        "state": state,
        "reason": reason,
        "valid_voxels": valid,
        "total_voxels": total,
    }


def _store_map_result(scan, model, data, result):
    """Attach input identity and record technical map finiteness in one place."""
    dim = "bvalue_idx" if model == "adc" else "echo_idx"
    result.attrs.update(
        fit_model_id=model,
        fit_reconstruction_type="seq2d",
        fit_dynamic_dim=dim,
        fit_dynamic_values=data.coords[dim].values.tolist(),
        fit_dynamic_unit="s/mm2" if model == "adc" else "s",
    )
    valid = int(np.count_nonzero(np.isfinite(result[model].values)))
    total = int(result[model].size)
    state = "failed" if valid == 0 else "computed" if valid == total else "partial"
    reason = f"No finite {model.upper()} values were produced." if valid == 0 else ""
    scan.fit_results[model] = result
    scan.fit_status[model] = _status(state, reason, valid, total)


def _fit_scanner_map(scan, model, adc_options, t2_options):
    """Select scanner images once, establish eligibility, then call analysis."""
    if not _supports_model(scan, model):
        scan.fit_status[model] = _status(
            "skipped", "Sequence is not supported for this map model."
        )
        return None
    data = scan.reconstructions.get("seq2d")
    if data is None:
        data = scan.reconstruct("seq2d")
    if data is None:
        scan.fit_status[model] = _status(
            "skipped", "No seq2d scanner images are available."
        )
        return None
    if not isinstance(data, xr.DataArray):
        raise TypeError("seq2d must be a labelled xarray.DataArray.")
    groups, visu, reason = _scanner_source(scan, data)
    if not reason:
        reason = (
            _adc_eligibility(scan, data, groups)
            if model == "adc"
            else _t2_eligibility(scan, data, groups, visu)
        )
    if reason:
        scan.fit_status[model] = _status("skipped", reason)
        return None
    result = fit_adc(data, adc_options) if model == "adc" else fit_t2(data, t2_options)
    _store_map_result(scan, model, data, result)
    return result


def _fit_scan_maps(scan, models, adc_options, t2_options=None):
    selected = _select_models(models)
    _validate_options(adc_options, t2_options)
    results = {}
    for model in selected:
        scan.fit_results.pop(model, None)
        scan.fit_status.pop(model, None)
    for model in selected:
        try:
            result = _fit_scanner_map(scan, model, adc_options, t2_options)
            if result is not None:
                results[model] = result
        except Exception as exc:
            context = (
                f"{model.upper()} map fitting ({type(scan).__name__}, seq2d): {exc}"
            )
            scan.fit_status[model] = _status("failed", context)
            exc.add_note(context)
            raise
    return results


def _fit_summary(scan):
    """List computed maps in English without data access or fitting."""
    statuses = vars(scan).get("_fit_status", {})
    parts = []
    for model, status in statuses.items():
        label = model.upper()
        state = status["state"]
        if state == "computed":
            parts.append(f"{label} computed")
        elif state == "partial":
            parts.append(
                f"{label} partially computed: {status['valid_voxels']}/{status['total_voxels']} voxels finite"
            )
    return "; ".join(parts)
