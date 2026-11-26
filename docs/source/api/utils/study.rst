Study
=====

Study and scan manipulation.

Standard scan summary
---------------------

``Study`` includes two additional columns in its DataFrame and styled display:

* ``Custom Reco`` follows ``Reconstructions`` and reports the latest HyperMRI
  ``raw`` reconstruction attempt. ``Success`` means that the result passed the
  strategy and geometry checks and was stored; it does not validate image
  quality or prove that the acquisition is complete. ``Failed`` includes a
  raised error or an unexpected ``None`` result. ``Not run`` means that no raw
  reconstruction was attempted or that it was explicitly disabled.
  ``No raw data`` means that the vendor reports no source for the sequence's
  raw reconstruction. Scanner images alone do not count as custom success.
* ``Measurement Time`` is always the last column, including after additional
  metadata columns are requested. It shows the acquisition clock time as
  ``HH:MM:SS`` in the timezone stored by the vendor, or ``MISSING``. For Bruker
  scans the source is ``acqp["ACQ_time"]``; the ``method`` file's
  ``MetaTimeStamp`` is not used because it can describe a later parameter edit.

Calling ``study.display()`` refreshes these columns from the current scan
state. A failed attempt remains visible even if an earlier successful raw
result is cached. A later successful reconstruction clears that failure.
Summary construction and refresh do not load samples or run reconstruction.

Scans that fail to load
-----------------------

``Study`` loads each scan with :func:`hypermri.load_scan`. When that raises,
for example because a reconstruction requested by ``load_data`` fails, the
scan is left out of ``study.dataframe``, the error is logged, and the
exception is stored in ``study.load_errors`` under the scan path; the other
scans still load. ``study.load_errors`` is empty when every detected scan
loaded. To inspect such a scan, load it with ``load_scan(path,
mode="lazy")`` and call ``scan.reconstruct(...)`` directly; a failed direct
attempt is then shown as ``Failed`` in ``Custom Reco``.

The colored display uses fixed status colors: ``Custom Reco`` success is green,
failure red, and missing raw data violet; ``Not run`` has no background color.
``DICOM`` uses green for ``Yes`` and red for ``No``. These colors do not depend
on which statuses occur in the study. Use ``study.display(colorize=False)`` to
disable background colors.

Bruker acquisition completion
-----------------------------

The relevant metadata are available through ``scan.handler.acqp``:

* ``ACQ_completed`` is the saved acquisition completion flag.
* ``ACQ_nr_completed`` and ``ACQ_scans_completed`` are the saved repetition
  and scan counters; ``ACQ_total_completed`` records the total scan count.
* ``ACQ_pipe_status`` records the acquisition pipeline state.

An affirmative ``ACQ_completed`` and ``ACQ_pipe_status="Wrapup"`` are not
sufficient evidence that every intended sample was acquired. The inspected
PV7 scans retain these values alongside partial-grid counters and conflicting
``acqp``/``method`` sizes. Compare the counters with the sequence's encoding
layout and requested repetitions, and verify that the raw file size agrees
with the declared storage format, receiver count and block padding. Missing
fields leave the completion state unknown. A universal completion classifier
is not implemented; sequence-specific acquisition layouts must be established
before labeling a scan complete or aborted.

.. autoclass:: hypermri.Study
   :members: display

Scan manipulation
-----------------

.. automodule:: hypermri.utils.study.manipulation
   :members:
   :undoc-members:
   :show-inheritance:
