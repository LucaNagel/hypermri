Time
====

Temporal processing and time axis calculations.

Sampling-axis input contract
----------------------------

``calc_sampling_time_axis`` requires one finite positive timing source:
``dwell_time_s`` in seconds or ``bw_hz`` (alias ``bandwidth_hz``) in Hz.
Supplying both is an error, even when their values are reciprocal. Omit an
unused source or pass ``None``; zero and negative placeholders are invalid.
There is no implicit timing default. For example::

   from hypermri.utils.time.temporal import calc_sampling_time_axis

   time_s = calc_sampling_time_axis(npoints=3, bw_hz=1000)
   # [0.0, 0.001, 0.002] seconds

Counts must be nonnegative integer scalars; an omitted context-free count
remains 512, while an explicit zero produces an empty axis. Do not supply both
``npoints`` and its alias ``no_points``, or both bandwidth aliases. Invalid
types raise ``TypeError``; conflicts, nonfinite/nonpositive timing and missing
timing raise ``ValueError``. This replaces the former accidental all-zero
bandwidth axis and the ``None > 0`` failure.
An explicit canonical zero count also means an empty axis with metadata;
replace former zero count sentinels with omission or ``None`` to derive the count.

With a metadata object, a matching valid cached sampling axis remains
authoritative, including irregular echo spacing and nonzero origins. Explicit
inputs are validated before using that cache. Otherwise one explicit timing
source controls spacing, or valid dwell metadata supplies it; omitted count
uses ``n_acquisition - cut_off``. Metadata-generated axes retain their historical
zero origin; ``start_with_0`` and ``t_echo_s`` affect only context-free axes.
The operation neither mutates metadata nor reconstructs data.

The mathematics compatibility entry point retains its historical 1 ms default
when both timing sources are omitted and its empty result for a zero count
without metadata. Its explicit inputs follow the same strict timing validation.
Migrate zero/negative timing placeholders to omission or valid measured values,
and remove redundant bandwidth arguments when supplying a dwell time.

.. automodule:: hypermri.utils.time.temporal
   :members:
   :undoc-members:
   :show-inheritance:
