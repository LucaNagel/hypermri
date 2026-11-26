"""Configure workers and dispatch independent indexed fitting tasks.

Scientific settings and calculations belong to the supplied callbacks. Importing
this module starts no workers; dispatch is eager and keeps no result cache.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, MutableMapping
from dataclasses import dataclass
from typing import TypeVar

from joblib import Parallel, delayed
from tqdm.auto import tqdm

EXECUTION_PARAM_KEYS = ("use_multiprocessing", "number_of_cpu_cores")

IndexT = TypeVar("IndexT")
ResultT = TypeVar("ResultT")


@dataclass(frozen=True)
class ExecutionConfig:
    """Store immutable worker settings shared by voxelwise fitting paths.

    Parameters
    ----------
    use_multiprocessing : bool, optional
        Request joblib execution when there is more than one task; default True.
        The active joblib backend determines whether processes or threads run.
    number_of_cpu_cores : int, optional
        Joblib worker count; -1 (default) uses all available CPUs, and 1 runs
        one worker. Other values follow joblib's ``n_jobs`` rules; zero is invalid.

    Notes
    -----
    Construction does not start workers or validate joblib settings. There is no
    mutable state or cache. Worker selection does not change scientific defaults.

    Examples
    --------
    >>> ExecutionConfig.from_mapping({"use_multiprocessing": False}).as_fit_params()
    {'use_multiprocessing': False, 'number_of_cpu_cores': -1}
    """

    use_multiprocessing: bool = True
    number_of_cpu_cores: int = -1

    @classmethod
    def from_mapping(cls, params: MutableMapping | dict | None) -> "ExecutionConfig":
        """Normalize the two worker keys from an existing parameter mapping.

        Parameters
        ----------
        params : MutableMapping or dict or None
            Mapping read without mutation. Missing keys use class defaults;
            a None worker count becomes -1. Other scientific keys are ignored.
            The parallel flag is converted with bool and worker count with int.

        Returns
        -------
        ExecutionConfig
            New immutable settings; no workers are started or values cached.

        Raises
        ------
        TypeError, ValueError
            If the supplied worker count cannot be converted to an integer.
        """
        params = params or {}
        number_of_cpu_cores = params.get("number_of_cpu_cores", -1)
        if number_of_cpu_cores is None:
            number_of_cpu_cores = -1
        return cls(
            use_multiprocessing=bool(params.get("use_multiprocessing", True)),
            number_of_cpu_cores=int(number_of_cpu_cores),
        )

    def as_fit_params(self) -> dict[str, bool | int]:
        """Return worker settings using the established fit-parameter key names.

        Returns
        -------
        dict
            New mapping containing ``use_multiprocessing`` and
            ``number_of_cpu_cores``; changing it does not mutate this config.
        """
        return {
            "use_multiprocessing": self.use_multiprocessing,
            "number_of_cpu_cores": self.number_of_cpu_cores,
        }


def copy_execution_params(
    source: MutableMapping | dict | None,
    target: MutableMapping,
) -> MutableMapping:
    """Copy supplied worker options into a caller-owned parameter mapping.

    Parameters
    ----------
    source : MutableMapping or dict or None
        Read-only source. Only the two keys in ``EXECUTION_PARAM_KEYS`` are
        considered; absent keys leave the target untouched. None is a no-op.
    target : MutableMapping
        Destination updated in place with the original values, without coercion.

    Returns
    -------
    MutableMapping
        The same target object. Scientific options and defaults are preserved;
        this deterministic operation starts no workers and caches nothing.
    """
    if source is None:
        return target
    for key in EXECUTION_PARAM_KEYS:
        if key in source:
            target[key] = source[key]
    return target


def run_indexed_tasks(
    index_iterator: Iterable[IndexT],
    task_func: Callable[[IndexT], ResultT],
    *,
    execution_config: ExecutionConfig,
    parallel_desc: str,
    sequential_desc: str,
) -> list[ResultT]:
    """Run independent indexed tasks either sequentially or with joblib.

    This helper centralizes the progress-bar behavior, joblib ``return_as``
    fallback, and sandbox/process-backend fallback used by voxelwise fitters.

    Parameters
    ----------
    index_iterator : Iterable
        Task identifiers, consumed once into a list before execution. Empty
        input returns an empty list without invoking the callback or joblib.
    task_func : callable
        Called once per identifier on the successful dispatch path. Owns all
        scientific values, units, precision and input-mutation requirements.
        Callbacks should be deterministic and safe to repeat after a backend
        startup failure; partially completed work can be rerun on fallback.
    execution_config : ExecutionConfig
        Worker selection. Serial execution is also used for a single task.
    parallel_desc : str
        Progress-bar label for the parallel path.
    sequential_desc : str
        Progress-bar label for serial execution, including backend fallback.

    Returns
    -------
    list
        Callback results without conversion or caching. Serial execution and
        older joblib retain input order; current parallel execution returns
        completion order. Include task identifiers in results when assembling
        labelled data. The call is eager and displays progress bars.

    Raises
    ------
    TypeError
        Propagates task/backend TypeError other than the legacy ``return_as``
        compatibility failure. Other callback exceptions also propagate.
    ValueError
        If joblib rejects worker settings. PermissionError emits RuntimeWarning
        and falls back to serial execution instead of propagating.

    Examples
    --------
    >>> sorted(run_indexed_tasks(
    ...     [2, 1], lambda index: (index, index**2),
    ...     execution_config=ExecutionConfig(False, 1),
    ...     parallel_desc="parallel", sequential_desc="serial",
    ... ))
    [(1, 1), (2, 4)]
    """
    index_list = list(index_iterator)
    if not index_list:
        return []

    if execution_config.use_multiprocessing and len(index_list) > 1:
        try:
            results = []
            with tqdm(total=len(index_list), desc=parallel_desc) as pbar:
                for result in Parallel(
                    n_jobs=execution_config.number_of_cpu_cores,
                    return_as="generator_unordered",
                )(delayed(task_func)(indices) for indices in index_list):
                    results.append(result)
                    pbar.update(1)
            return results
        except TypeError as exc:
            if "return_as" not in str(exc):
                raise
            try:
                return Parallel(n_jobs=execution_config.number_of_cpu_cores)(
                    delayed(task_func)(indices) for indices in index_list
                )
            except PermissionError as fallback_exc:
                warnings.warn(
                    "Parallel voxel fitting could not start because the process "
                    f"backend is unavailable ({fallback_exc}). Falling back to "
                    "sequential fitting.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except PermissionError as exc:
            warnings.warn(
                "Parallel voxel fitting could not start because the process "
                f"backend is unavailable ({exc}). Falling back to sequential fitting.",
                RuntimeWarning,
                stacklevel=2,
            )

    return [task_func(indices) for indices in tqdm(index_list, desc=sequential_desc)]
