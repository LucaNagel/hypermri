CSI fitter validation
=====================

This page is the single record of the scientific evidence behind the
spectroscopy fitter, its noise conventions and the Q0/Q1/Q2 quality rule.
The previous tutorial collection and its figures have been retired; the
scientific results and their provenance remain here.

All results below come from synthetic studies.  They describe the simulated
signal model and search settings listed here and are not universal detection or
quantification limits for measured data.

Study design
------------

Signal model
^^^^^^^^^^^^

Each spectrum is a complex free induction decay (FID) with five Lorentzian
components (pyruvate, lactate, pyruvate hydrate, alanine and bicarbonate) and
white, zero-mean, circular complex Gaussian noise.  Noise ``sigma`` is the
standard deviation of one quadrature (real or imaginary part) and is known
exactly unless a noise-estimation case says otherwise.

.. list-table:: Frozen acquisition and fit settings
   :header-rows: 1
   :widths: 40 60

   * - Setting
     - Value
   * - Samples, dwell time
     - 256 complex samples, 0.4 ms
   * - Component frequencies (Hz)
     - −69.75, 856.83, 566.19, 339.71, −830.41
   * - Background amplitudes (relative)
     - 1.0, 0.8, 0.1, 0.05, 0.02
   * - Initial T2*
     - 25 ms for all components
   * - Frequency search width
     - 70 Hz
   * - T2* search width and bounds
     - 100 ms within 1–160 ms
   * - Grid search
     - ``npoints=31``, ``niter=1``, ``rep_fitting=11``, ``zoomfactor=1.5``
   * - Amplitude solver
     - ``pinv`` without ridge regularisation, no amplitude filter
       (``snr_threshold=None``)
   * - Minimum peak separation
     - 5 Hz
   * - Seed of the study design
     - 202609080

These are study settings, not defaults for every measurement.

The FID-SNR of a target is ``abs(a_true(0)) / sigma``: the true complex
amplitude at the model time origin divided by the quadrature sigma.  It is not a
peak height read from a magnitude spectrum.  The Fit-SNR is
``abs(a_hat) / s_a``, where ``s_a`` is the linear standard deviation of one
amplitude quadrature with the fitted frequencies and T2* held fixed.  For an
isolated peak, Fit-SNR ≈ FID-SNR · sqrt(sum exp(−2 t/T2*)); with the settings
above and T2* = 25 ms this factor is about 5.62, as observed in the study.
Other T2*, sampling or peak overlap change the factor.

Stages
^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage
     - Design
   * - 0: point accuracy
     - Target FID-SNR 1, 2, 4, 8, 10, 30 and 100 at T2* = 25 ms, plus 5 ms and
       80 ms at FID-SNR 2; 30 replicates per target and condition.  No model
       comparison and no bootstrap.
   * - 1: weak-peak detection
     - M0/M1 parametric bootstrap at α = 0.05 with B = 39 null simulations per
       spectrum.  Target FID-SNR 0 (null), 1, 2, 4 and 8 with 20 replicates and
       strong anchors at 10 and 20 with 10 replicates; 100 null evaluations per
       target; T2* = 25 ms.
   * - 2: parameter uncertainty
     - Full-refit M1 bootstrap with B = 49 draws and 95 % percentile intervals.
       Target FID-SNR 1, 2, 4 and 10 at T2* = 25 ms, plus 5 ms and 80 ms for
       lactate and bicarbonate at FID-SNR 2; 20 outer replicates.
   * - 3: noise sources
     - Direct noise estimation (40 replicates per case; 64 noise samples,
       sigma = 1) for known sigma, matched RF-off, noise-only dynamic frames,
       outside-object CSI masks and contaminated variants (1, 5 and 10 %
       residual signal).  Automatic outer-k-space selection (40 replicates per
       case; 400 reference cases in each of the training, calibration and
       validation partitions; minimum 12 selected points, at most 5 % signal
       energy, minimum noise-dominance probability 0.95; diagnostic samples
       0–31, estimation samples 32–63).  End-to-end detection with estimated
       sigma.
   * - 4: renal multi-component models
     - Designed (one to three Z-OMPD C5 compartments, 172 conditions) but not
       run as a scientific full run.
   * - Final confirmation
     - Designed and implemented but not run.

Results
-------

Data basis
^^^^^^^^^^

The evaluated full runs are ``89356503391effdc`` (stages 0–2) and
``b132584eed7d43f5`` (stage 3).  The incomplete stage-3 run
``546e70bb2b4f0aaa``, technical smoke runs and the earlier Voigt/18 Hz stage-4
artifacts ``a04442afabf02caf`` and ``a98333236bf9f4b1`` are not evidence; the
latter are incompatible with the later Lorentzian renal design.  The original
integrity check found matching manifest and result counts, no duplicate result
keys and no missing referenced test data.

.. list-table:: Scope of each stage
   :header-rows: 1
   :widths: 20 35 45

   * - Stage
     - Scope
     - Result
   * - 0
     - 45 conditions, 1,350 spectra
     - Joint amplitude, frequency and T2* accuracy rises clearly with SNR.
       Some practical error limits were chosen after the run.
   * - 1
     - 1,000 spectra (500 null, 400 weak alternatives, 100 anchors);
       39,000 null simulations
     - 31/500 false alarms (6.2 %, exact binomial 95 % interval 4.3–8.7 %).
       98/100 targets detected at FID-SNR 1 and all from FID-SNR 2.  No null
       case reached Fit-SNR 10.
   * - 2
     - 480 outer spectra, 23,520 bootstrap fits
     - All fits succeeded; intervals narrow with SNR.  Nominal 95 % coverage
       is not confirmed.
   * - 3
     - 720 direct noise cases, 400 k-space cases, 660 end-to-end spectra,
       25,740 null simulations
     - Matched explicit references work well in these conditions.  Sigma
       underestimation raises false alarms markedly.

In stage 1, alanine had 10/100 false alarms, which is not enough to support a
separate threshold.  All 83 frequency-bound hits occurred in null cases.

Detection and joint accuracy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A fit meets the joint accuracy criteria when the relative amplitude error and
the relative T2* error are at most 20 % and the frequency error is at most
5 Hz.  T2* limited joint accuracy most often.

.. list-table::
   :header-rows: 1
   :widths: 10 22 14 18 16 20

   * - FID-SNR
     - Joint criteria met (stage 0, T2* = 25 ms)
     - Median Fit-SNR (stage 2)
     - Median amplitude interval width / truth
     - Median frequency interval width
     - Median T2* interval width / truth
   * - 1
     - 40/150 (26.7 %)
     - 6.04
     - 89.4 %
     - 9.69 Hz
     - 207.6 %
   * - 2
     - 100/150 (66.7 %)
     - 11.50
     - 45.3 %
     - 4.27 Hz
     - 70.3 %
   * - 4
     - 149/150 (99.3 %)
     - 22.55
     - 22.4 %
     - 2.00 Hz
     - 32.7 %
   * - 10
     - 150/150 (100 %)
     - 56.33
     - 9.1 %
     - 0.83 Hz
     - 12.8 %

Widths are full interval widths, not ± errors.  From FID-SNR 8 every 25 ms
group of stage 0 had 150/150 successes; 100 % in a finite sample is not a
guarantee.

Run ``89356503391effdc``.  Detection rises at lower SNR than joint
parameter accuracy.  The Fit-SNR lines 10 and 20 mark the two working
thresholds derived from these results.

Thresholds of the quality rule
""""""""""""""""""""""""""""""

``evaluate_spectroscopy_fit_quality`` uses two provisional thresholds derived
from stages 0–2:

* **Fit-SNR 10:** with a valid fit and a suitable noise reference, a peak may
  be classified as provisionally detected without the full M0/M1 test.  Below
  10 the test is required.  No null case of stage 1 reached this value.  The
  threshold alone gives no p-value.
* **Fit-SNR 20:** joint quantitative interpretation may be considered when
  valid uncertainty information is also available and no other Q2 block
  applies.  Median Fit-SNR 22.55 at FID-SNR 4 coincides with 149/150 fits
  meeting the joint criteria.

The condition-number warning at 20 lies above the maximum of 16.82 observed in
the full runs; it is not a separately calibrated physical limit.

Parameter intervals
^^^^^^^^^^^^^^^^^^^

The descriptive overall coverage of the stage-2 intervals was 93.8 % for the
amplitude magnitude, 90.4 % for the frequency and 89.6 % for T2*.  With
linear 2.5 % and 97.5 % quantiles at B = 49, the order statistics that define
the interval enclose about 91.2 % of the central rank mass.  This is a finite
rank resolution, not a guarantee of the actual coverage.  The intervals must
not be reported as validated 95 % intervals.

Conditional coverage and widths from stage 2.  The confidence intervals on
the coverage rates describe the uncertainty of these observed rates; they
do not correct the bootstrap intervals.  Each peak and condition has only
20 outer replicates, so non-monotonic coverage rates are not a reliable SNR
effect; the decreasing widths are clearer.

Short T2* stays difficult.  At FID-SNR 2 and T2* = 5 ms, lactate and
bicarbonate had median frequency widths of 57.6 Hz and 48.6 Hz and relative
T2* widths of 434 % and 326 %.  At T2* = 80 ms the Fit-SNR was about 19.3,
but the relative T2* width was still 56–65 %.  A numerically successful fit is
not enough for an accurate T2* in these cases.

Example spectra
"""""""""""""""

Stored stage-1 FIDs; for each SNR the alanine replicate with the Fit-SNR
closest to the group median.  At FID-SNR 2 this is replicate 7 with
Fit-SNR 11.08 and p = 0.025.  The FFT is zero-filled from 256 to 8,192
points without apodisation; zero filling smooths the display but adds no
information.  The noise-free reference uses the frozen true parameters.

Noise estimation
^^^^^^^^^^^^^^^^

For the synthetic RF-off reference, the mean ratio of estimated to true sigma
was 1.00024 with a relative RMSE of 0.61 %.  A truly signal-free region
outside the object gave 0.99953 and 0.75 %.  All eight matching empirical
noise modes gave the same binary detection decisions as known sigma in the
end-to-end cases tested.  A static slice/PRESS signal without a reference
gave no estimate.

An underestimate of sigma by 20 % produced 10 false alarms in 40 null cases;
with the correct sigma there were 0 of 40.  At a fixed fit, using
``c · sigma`` scales the Fit-SNR by ``1/c`` and the M0/M1 statistic by
``1/c²``; the p-value cannot be converted the same way because the simulated
null distribution also depends on sigma.

Run ``b132584eed7d43f5``.  Matching explicit references are nearly
unbiased.  The automatic outer-k-space selection rejected the tested
temporal correlation and non-circular noise but accepted 20/40 cases with
10 % signal contamination spread over the outer k-space. A deliberate sigma
underestimate changes detection as described above.

The automatic outer-k-space selection (``estimate_csi_noise``) is therefore
not a general replacement for a missing reference measurement.  The end-to-end
samples were small and, for present targets, limited to bicarbonate at
FID-SNR 2.  The observed conservative effect of a 20 % sigma overestimate does
not show that weak peaks are not lost.

Limitations
-----------

The results above do not cover the following; each needs its own study before
a claim is made.

* **Model misspecification.** All studies simulate the fitted Lorentzian
  model with the correct number of components.  Missing or extra components,
  non-Lorentzian line shapes, baseline or phase errors are not covered.
  Overlapping components of the same resonance, as in renal compartments, were
  not evaluated (stage 4 was not run).
* **Correlated noise.** The detection test, the bootstrap and the local
  covariance assume white, circular noise with a known quadrature sigma.  The
  reconstructed CSI noise covariance is not yet connected to a
  covariance-weighted detection test; a reconstructed marginal sigma must not
  be used as a known white sigma.  Spatial correlation affects ROI sums and
  joint statements over voxels.
* **Amplitude-magnitude bias.** ``abs(a_hat)`` is biased upwards at low SNR
  because noise adds to the magnitude.  A fit returns a non-zero magnitude even
  for pure noise.  Percentile and local intervals of the magnitude are not
  calibrated near zero.
* **Unidentified frequency and T2* for absent peaks.** When the true target
  amplitude is zero, its frequency and T2* are not identifiable; intervals for
  them have no meaning.  Bound hits of a weak peak can invalidate the joint
  covariance of the whole spectrum.
* **Multiple testing.** ``p <= 0.05`` is an uncorrected single-test decision.
  Many voxels or targets need a pre-specified correction; no false-discovery
  rule has been validated.  The stage-1 false-alarm rate of 6.2 % (95 %
  interval 4.3–8.7 %) does not demonstrate the nominal 5 %; the parametric
  bootstrap with plugged-in M0 estimates does not guarantee an exact error
  rate.
* **Bootstrap Monte Carlo resolution.** With B = 39 the smallest possible
  p-value is 0.025, a coarse resolution near α = 0.05.  With B = 49 the
  percentile intervals have an effective central rank mass of about 91 %.
  ``detection_mc_tail_low`` and ``detection_mc_tail_high`` describe only the
  Monte Carlo uncertainty of the simulated exceedance fraction.
* **External validity.** No measured data with known truth were evaluated.
  The Fit-SNR thresholds 10 and 20 have not been confirmed independently.

Provenance and reproduction
---------------------------

The study runner, design files and figure script lived in
``experiments/spectroscopy_peak_quality_v4``.  They were removed from the
repository on 2 October 2026; commit ``52de066f`` is the last revision that
contains them.  The run artifacts (manifests, ``results.csv``,
``interval_results.csv``, ``bootstrap_draws.csv`` and the stored stage-1 FIDs)
were never in Git and are kept outside the repository by the owner.  Each
run's manifest records its frozen design and source state, which differ from
the later design extended by stage 4 and the final confirmation.

The four retired study figures in ``docs/source/_static/spectroscopy`` were
produced from these artifacts without new fits by
``experiments/spectroscopy_peak_quality_v4/plot_final_fit_quality_evaluation.py``
at that revision.  ``noise_quadrature_and_tail.png`` is a didactic figure of
the retired noise-quantification notebook and is not study evidence. The owner
authorized removal of these figures with all old tutorial media on
10 October 2026; their retirement does not change the numerical results above.

``scripts/simulate_13c_spectroscopy.py`` remains as an exploratory simulation
of five 13C metabolites with the local covariance of
``estimate_fit_covariance``.  Its default of 30 repetitions is not a coverage
validation.  Its tests are in ``tests/validation``.

Unit tests check program behaviour.  False-alarm rate, power, parameter error
and interval coverage need separate studies such as the ones above.
