Utilities
=========

The utilities package provides a comprehensive set of helper functions and classes
organized by domain.

.. toctree::
   :maxdepth: 2
   :caption: Utility Modules:

   utils/mathematics
   utils/physics
   utils/image
   utils/mri
   utils/spectroscopy
   utils/io
   utils/system
   utils/coordinates
   utils/time
   utils/sequence
   utils/study

Static plotting and slice matching
----------------------------------

Static colorbars, colormaps and subplot layout helpers live in
``hypermri.utils.plot.matplotlib_helpers``. The former ``utils_plot`` module
was removed on 15 November 2026. These helpers retain
their established Matplotlib behavior; use the common viewer for interactive
scientific visualization.

.. automodule:: hypermri.utils.plot.matplotlib_helpers
   :members: add_colorbar, plot_3d, subplot_auto_size, remove_empty_subplots, format_y_tick_labels, create_colormap

Slice-position matching is independent of static rendering and belongs to
``hypermri.utils.image.slices``. Existing nearest-centre matching, strict
tolerance and metadata precedence are unchanged. Analysis callers import this
function directly and therefore do not initialize Matplotlib layout helpers.

.. autofunction:: hypermri.utils.image.slices.get_overlapping_slices

Backward Compatibility
----------------------

The legacy ``hypermri.utils.utils_general`` and ``hypermri.utils.utils_logging``
modules no longer exist as standalone modules. Their functionality has been
split into the domain-specific modules listed above, especially
``hypermri.utils.system`` and the subpackages under ``hypermri.utils``.
