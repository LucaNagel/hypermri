Worked Example: Adding Two Balanced-SSFP Sequences
==================================================

This page walks through adding new sequences to the reconstruction
pipeline, from the scanner method string to a tested reconstruction.
:doc:`adding_sequence` is the reference for each mechanism; this page shows
the order in which they are used and the decisions on the way.

The examples are two balanced-SSFP sequences used for deuterium metabolic
imaging (DMI) by the Frydman group at the Weizmann Institute: a
density-weighted CSI and a multi-echo SSFP imaging sequence. Their MATLAB
reconstruction and mouse data sets are published on the group's software
page as ``pro_DMI_SSFP_RK_SK_SpecRecon``. The data are not part of
HyperMRI; keep any local copy outside Git.

Step 1: Look at the Data Before Writing Code
--------------------------------------------

Read the parameter files of one scan as text. For this sequence they show:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Parameter
     - Value and meaning
   * - ``Method`` (``method``)
     - ``<User:ssfpCSI>``, a custom ParaVision 6.0.1 method.
   * - ``PVM_Nucleus1``
     - ``2H``; working frequency 99.76 MHz, offset 2.5 ppm.
   * - ``PVM_Matrix``, ``PVM_EncSteps0/1``
     - 32 × 32 phase-encoded positions, ``LINEAR_ENC LINEAR_ENC``, one
       slice.
   * - ``PVM_DigNp``, ``PVM_DigSw``
     - 48 complex points per FID at 5000 Hz spectral width.
   * - ``AverageList``
     - 1024 entries, one per k-space position, between 1 and 8: a
       density-weighted acquisition (scan name ``CSI-bSSFP_wtd``). The
       sum is 2775 FIDs per repetition.
   * - ``PVM_NRepetitions``
     - 12 repetitions of the whole k-space.
   * - ``ACQ_jobs``, ``rawdata.job0``
     - 33 300 FIDs = 2775 × 12, little-endian int32 pairs. The ``fid`` file
       has a different size and is not used.
   * - ``pdata/1/2dseq``
     - The scanner reconstruction, 48 × 32 × 32 × 12 frames. It is the
       reference for the new reconstruction.

The published MATLAB reader reads ``rawdata.job0``, reshapes it to
``(points, sum(AverageList), repetitions)``, adds the repeated FIDs of each
k-space position in the order of ``AverageList`` and reshapes the result to
the matrix. Treat such code as a description of the acquisition, not as the
reference: the scanner reconstruction ``2dseq`` is the reference.

Step 2: See What HyperMRI Does Today
------------------------------------

Load the study and reconstruct one scan:

.. code-block:: python

   from hypermri import Study

   study = Study("csi_ssfp_data_mouse_tumor_pancreas", verbose=False)
   study.dataframe[["Sequence Type", "Scan Name"]]
   study[8].reconstruct("raw")

The method string contains ``csi``, so the ``contains:csi`` pattern of the
``csi`` entry in ``sequence_mapping.yaml`` makes the scan a
:class:`~hypermri.sequences.CSI`. Before this sequence was added, the raw CSI
reconstruction stopped with ``The CSI encoding order is not defined for
Method='<User:ssfpCSI>'``. This is the intended behaviour: the CSI strategy
only accepts methods whose encoding it has verified, and it never guesses.

Step 3: Decide How Much Is New
------------------------------

Ask what actually differs from a sequence HyperMRI already reconstructs:

* **Only the method name:** add one ``exact:`` alias in
  ``sequence_mapping.yaml``.
* **The acquisition order or loops:** keep the sequence ID and class, and
  teach the existing strategy the new order in one named place.
* **The data or the processing:** add a new canonical ID, a sequence class
  and a strategy (see :doc:`adding_sequence`).

For the density-weighted CSI the geometry, the spectral axis, the repetitions
and the storage format are those of an ordinary CSI. Only the encoding
differs: position *i* of the ``PVM_EncSteps0/1`` grid is acquired
``AverageList[i]`` times in a row. The sequence therefore keeps the ``csi``
ID and the :class:`~hypermri.sequences.CSI` class, and the CSI strategy
learns this order. The method still gets an ``exact:`` line under ``csi``:
every CSI method the reconstruction supports is named explicitly in the
mapping, and a test checks that both lists agree.

Step 4: Describe the Encoding Once
----------------------------------

The CSI strategy separates *where* each FID belongs from *how* the FIDs are
put on the grid. ``get_phase_encoding_order`` returns one read and one phase
position per acquired FID; the sorter
:func:`~hypermri.reconstruction.csi_raw.sort_csi_encoding_grid` places them
and counts how often each position was acquired. For the new method, the
strategy takes the positions of the ``PVM_EncSteps0/1`` grid and repeats each
one by its ``AverageList`` entry:

.. code-block:: python

   # in reconstruction/strategies/bruker/csi.py
   if method_name in DENSITY_WEIGHTED_METHODS:
       order, read, phase = _scanner_encoding_positions(handler)
       return (order, *_repeat_by_average_list(handler, read, phase))

``_repeat_by_average_list`` checks that ``AverageList`` holds one positive
integer per position and names the method in the error otherwise. The method
is also listed in ``FOURIER_ENCODED_METHODS`` in ``csi_raw.py``, so its grid
is recognised as k-space. Nothing else changes: the spatial Fourier
transform, the field-of-view shift, the coordinates and the precision are the
ones of every other CSI.

Step 5: Let the Scanner Decide Open Questions
---------------------------------------------

The sorter can combine repeated FIDs of one position in two ways: average
them or add them. The MATLAB script adds them, which keeps the density
weighting in the data. The comparison with ``2dseq`` settles which one the
scanner does: with averaging the ``raw`` result agrees with ``2dseq`` to a
correlation of 0.99999999 and a relative residual of 6e-5 (float32
precision); with adding the correlation drops to 0.69. The same comparison
shows that no image or spectrum flips are needed.

The default is therefore ``"mean"``. Adding stays available as an explicit
option for analyses that want the density weighting:

.. code-block:: python

   averaged = scan.reconstruct("raw")
   weighted = scan.reconstruct("raw", repeated_positions="sum")

The option is listed in the strategy's ``accepted_options`` and checked
before any data are read. The noise diagnostics refuse a summed grid, because
its noise level differs between positions.

Step 6: A Second Example, Multi-Echo SSFP
-----------------------------------------

The data set ``me_ssfp_data_mouse_tumor_pancreas`` uses the method
``<User:nmrsuFISP_H2_Fb>``. Its name contains ``FISP``, so it already maps
to ``bssfp`` and the bSSFP strategy. Each scan has one slice and several
echoes; their times are the method parameter ``DelayArrayForTE`` (ms), and
each echo is stored as one acquisition object. The reader
``brukerapi.dataset.Dataset`` returns them in the object axis with the echo
as the outer loop, which matches ``2dseq`` echo for echo (correlation 1.0).

The bSSFP strategy already handles objects as movie frames. For this method
the objects become a labelled echo dimension instead:

* ``MULTI_ECHO_METHODS`` in ``reconstruction/strategies/bruker/bssfp.py``
  lists the method;
* :func:`~hypermri.reconstruction.strategies.bruker.bssfp.multi_echo_times_s`
  converts ``DelayArrayForTE`` to seconds and checks that there is one
  finite, positive, increasing time per echo;
* the result has the dimension ``echo_idx`` with the echo times in s as
  coordinate; several slices or movie frames are refused with a pointer to
  ``scan.reconstruct("seq2d")``, because that layout is not verified;
* :meth:`~hypermri.sequences.BSSFP.get_nechoes` counts the echo times.

Separating the deuterium species from the echoes is an analysis step on top
of this result, not part of the reconstruction.

Step 7: Test It
---------------

Use small synthetic data for the routine tests; never copy the downloaded
scans or their metadata into the repository.

* **Mapping:** ``<User:ssfpCSI>`` resolves to ``csi`` and
  ``<User:nmrsuFISP_H2_Fb>`` to ``bssfp``.
* **Encoding:** on a 3 × 2 grid an ``AverageList`` such as
  ``[1, 3, 2, 1, 1, 2]`` repeats each position by its count, and a list of the
  wrong length, with zeros or with fractions raises ``ValueError``.
* **Combination:** two FIDs at one position are averaged by default and
  added with ``repeated_positions="sum"``; an unknown rule raises.
* **Echoes:** invented echo labels arrive in ``echo_idx`` in order, with the
  echo times in s; inconsistent ``DelayArrayForTE`` values raise.
* **Repeatability:** two reconstructions of the same handler are equal, and
  the raw samples are unchanged.
* **Reference (opt-in):** with the real data available locally, the ``raw``
  result agrees with ``2dseq`` within the stated tolerance. Keep such checks
  outside the repository.

Finally run the narrow tests, then the repository gates listed in
:doc:`testing`.

Checklist
---------

#. Read ``method``, ``acqp`` and, if available, the vendor or author code.
#. Load the study and note what HyperMRI does now.
#. Find what differs: name, acquisition order, or data and processing.
#. Put a new acquisition order in one named place of the existing strategy;
   add a new ID, class and strategy only for new data or processing.
#. Compare with ``2dseq`` before choosing defaults, and keep alternatives as
   explicit options.
#. Synthetic tests for mapping, numerics and contract; an opt-in comparison
   against the scanner reconstruction.
