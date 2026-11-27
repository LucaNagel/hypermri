Viewer development
==================

How the browser viewer is built and how to extend it. Read
:doc:`../concepts/viewer` first for the split between Python and the browser.

Where code belongs
------------------

Every viewer feature has up to four parts, each in its own place:

.. list-table::
   :header-rows: 1
   :widths: 28 40 32

   * - Part
     - Content
     - Location
   * - Scientific operation
     - fits, mask projection, noise estimation; no viewer imports
     - ``hypermri.analysis``
   * - Viewer data
     - what a panel shows: planes, curves, overlays, units, geometry; plain
       functions on xarray data, testable without a browser
     - ``hypermri.visualization`` (for example ``fit_tool.py``,
       ``render_data.py``, ``geometry.py``)
   * - Python half of the view
     - traits sent to the browser, answers to browser messages, public
       methods such as ``set_frequency_window``
     - ``hypermri.visualization.scan_view``
   * - Browser half
     - drawing with Plotly.js and the controls; no scientific decisions
     - ``hypermri/visualization/browser/*.js``

A tool callback in the browser sends a message; the Python half calls a
function of the viewer-data layer, which calls the analysis function, and
sends the result back. The browser never computes a fit or a mask projection.

Layout
------

:mod:`hypermri.visualization.layout` owns what appears and where:

* ``PANELS``: the plots (``image``, ``curve``, ``time``);
* ``TOP_BAR``, ``OPTION_SECTIONS`` and ``BELOW_PLOTS``: which control groups
  sit in the top bar, in the "Plot options" panel and below the plots;
* ``TOOLS``: the tools in the tool menu, shown one at a time;
* ``BROWSER_PARTS``: the JavaScript files, joined in this order into the one
  script the notebook loads.

Moving a control group means moving its name between these tuples; the
browser builds exactly the listed groups. There is no JavaScript build step:
the files are plain JavaScript sharing one module scope.

Adding a tool
-------------

1. Write the scientific operation in ``hypermri.analysis`` with a NumPy
   docstring and tests on synthetic data.
2. Write the tool's settings and results as plain functions in a
   ``hypermri/visualization/<name>_tool.py`` module (compare
   ``ideal_tool.py``): settings as a small dataclass, one function to change
   an option, one to run, one to turn the result into what the browser draws.
3. Add the browser controls as ``browser/controls_<name>.js``. Controls send
   ``model.send({type: "<name>", action, ...})`` and redraw from the state the
   Python half sends back.
4. In ``scan_view``, answer the message in
   ``ScanView._handle_custom_message`` by delegating to the functions of
   step 2, and report a ``ValueError`` in the tool instead of raising it.
5. List the control group in ``CONTROL_GROUPS`` and ``TOOLS`` and the file in
   ``BROWSER_PARTS``.

Saved snapshots
---------------

:meth:`~hypermri.visualization.scan_view.ScanView.to_html` writes the state
of a view with the same browser code into one HTML file;
``browser/snapshot.js`` stands in for the Python half. Control groups listed
in ``PYTHON_GROUPS`` in ``browser/viewer.js`` are switched off there; a new
control group that needs Python belongs in that list, and a single input in an
otherwise browser-only group is marked with ``needsPython``. The anatomy slice
is the one Python answer saved for every choice: ``to_html`` stores the
changed traits per slice as ``_anatomy_slices``, and the snapshot model shows
them when the slider sends its controls. The documentation's viewer frames
are written by ``scripts/build_docs_snapshots.py``; run it after a visible
change to the viewer.

Tests
-----

* Viewer data and tool functions: unit tests without a widget, for example
  ``tests/unit/visualization/test_ideal_tool.py``.
* Layout: ``test_browser_layout.py`` checks that every listed group and file
  exists.
* Python half: ``test_scan_view.py`` constructs ``ScanView`` objects and sends
  them browser messages, without a browser.
* Snapshots: ``test_scan_view_snapshot.py``.

The browser half itself is checked by hand in JupyterLab and VS Code, and in
a saved snapshot opened in a browser.
