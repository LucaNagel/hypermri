Adding Sequences, Aliases, Reconstructions, Vendors, and Parameters
===================================================================

This guide describes the current registry-driven loading flow.
:doc:`sequence_walkthrough` applies it step by step to new sequences. The important
change is that sequence classes no longer decide their automatic
reconstruction in ``__init__``. Sequence classes should describe sequence
behavior; ``Study`` and ``BaseSequence.run_default_reconstructions`` decide
which reconstructions run for ``load_data=False``, ``"minimal"``, or ``True``.

What Happens When ``Study(path)`` Loads a Scan
----------------------------------------------

1. ``Study`` scans the study directory and calls
   :func:`hypermri.loading.scans.load_scan` for each measurement.
2. ``load_scan`` detects the vendor through :mod:`hypermri.vendors.registry`
   and constructs the matching vendor handler once. Bruker scans usually
   become ``BrukerHandler`` instances.
3. The vendor handler reads metadata and exposes normalized parameters through
   ``handler.get_parameters()``.
4. The sequence registry in :mod:`hypermri.sequences.registry` reads
   ``src/hypermri/sequences/config/sequence_mapping.yaml`` and maps the raw
   scanner method string, for example ``<Bruker:RARE>``, to a canonical ID.
   Maintained IDs resolve through ``sequences/_sequence_types.py``; explicit
   custom ``class`` paths retain the compatibility import route.
5. The selected sequence class is instantiated with that handler.
   ``BaseSequence`` initializes the experiment state from it and does not
   create a second handler. A scan without a registered sequence class becomes
   a generic :class:`hypermri.loading.experiment.MRIExperiment` built from the
   same handler. Sequence classes accept only the handler and the documented
   construction keywords; an ``MRIExperiment`` or an unknown keyword raises a
   ``TypeError``.
6. ``Study`` calls ``sequence.run_default_reconstructions(...)``. The default
   policy is:

   * ``load_data=False`` or ``"lazy"``: do not reconstruct immediately.
   * ``load_data="minimal"``: anatomical/diffusion sequences load ``seq2d``;
     spectroscopy sequences load ``raw``.
   * ``load_data=True``: anatomical/diffusion sequences load ``seq2d`` and
     ``raw``; spectroscopy sequences load ``raw``.

7. Each reconstruction call uses
   :func:`hypermri.reconstruction.selection.find_reconstruction`. It looks up the
   exact sequence key, then the canonical sequence key, then the vendor
   default in :data:`~hypermri.reconstruction.selection.RECONSTRUCTIONS`.
   ``reconstruct`` rejects options outside the class's ``accepted_options``,
   merges the caller's options over the defaults, and calls
   ``strategy_class().reconstruct(handler, **options)``.

Adding a Sequence Alias
-----------------------

Use this when a scanner method is really an existing sequence variant. For
example, ``<User:my_flash_variant>`` may still be a FLASH acquisition.

Edit only ``src/hypermri/sequences/config/sequence_mapping.yaml`` and add an
``exact:`` pattern under the existing canonical sequence key.

.. code-block:: yaml

   flash:
     category: anatomical
     method_patterns:
       - "exact:<Bruker:FLASH>"
       - "exact:<User:my_flash_variant>"
       - "contains:flash"

The same mapping is also used by reconstruction alias resolution, so
``<User:my_flash_variant>`` will resolve to the canonical key ``flash`` when
``MRIExperiment.reconstruct(...)`` asks for a reconstruction strategy.

Maintained classes belong to the explicit Python sequence type mapping. Custom
files may still name a dotted ``class`` path to override a maintained type or
introduce a custom ID. Registering a Python type at runtime remains supported.

Adding a New Custom Sequence
----------------------------

Use this when the sequence needs its own public class or helpers. A simple
single-file sequence is still valid, but a directory package is preferred once
the sequence needs several helper modules.

Minimal single-file layout:

.. code-block:: python

   # src/hypermri/sequences/MySequence.py
   from hypermri.loading.protocols import VendorHandlerProtocol

   from .base_anatomy import BaseAnatomy


   class MySequence(BaseAnatomy):
       """Short description of the sequence."""

       def __init__(self, handler: VendorHandlerProtocol, **kwargs):
           super().__init__(handler, **kwargs)
           # Add sequence-specific metadata or helpers here.
           # Do not call self.reconstruct(...) here.

Directory layout for a more complex sequence:

.. code-block:: text

   src/hypermri/sequences/my_sequence/
   ├── __init__.py
   ├── sequence.py
   ├── parameters.py
   └── compute.py

The public class can stay small:

.. code-block:: python

   # src/hypermri/sequences/my_sequence/__init__.py
   from .sequence import MySequence

   __all__ = ["MySequence"]

.. code-block:: python

   # src/hypermri/sequences/my_sequence/sequence.py
   from hypermri.loading.protocols import VendorHandlerProtocol

   from ..base_anatomy import BaseAnatomy
   from .parameters import collect_my_sequence_parameters


   class MySequence(BaseAnatomy):
       def __init__(self, handler: VendorHandlerProtocol, **kwargs):
           super().__init__(handler, **kwargs)
           self.my_sequence_parameters = collect_my_sequence_parameters(self)

Add its canonical ID and imported Python type to
``sequences/_sequence_types.py``, then add its category and method patterns to
``sequence_mapping.yaml``. The mapping file names no Python classes:

.. code-block:: yaml

   my_sequence:
     category: anatomical
     method_patterns:
       - "exact:my_sequence"
       - "exact:<Bruker:MySequence>"
       - "contains:my_sequence"

A class kept outside the package is registered in Python instead, before the
data are loaded:

.. code-block:: python

   from hypermri.loading.contracts import SequenceCategory
   from hypermri.sequences import register_sequence

   register_sequence(
       "my_sequence",
       MySequence,
       SequenceCategory.ANATOMICAL,
       ["exact:<Bruker:MySequence>"],
   )

Choose the category from ``anatomical``, ``spectroscopy``, ``diffusion``,
``perfusion``, or ``functional``. The category affects the default load policy
when the class does not define its own ``RECONSTRUCTION_POLICY``.

Adding a New Reconstruction
---------------------------

Reconstruction logic lives in small modules under
``src/hypermri/reconstruction/strategies/``.

Create a strategy class by subclassing ``BaseRecoStrategy``:

.. code-block:: python

   # src/hypermri/reconstruction/strategies/bruker/my_sequence.py
   import xarray as xr

   from hypermri.reconstruction.strategies.base import BaseRecoStrategy


   class BrukerMySequenceRecoStrategy(BaseRecoStrategy):
       accepted_options = frozenset({"data_precision", "no_reconstruction"})

       def reconstruct(self, handler, **kwargs) -> xr.DataArray:
           # Read data from handler, apply reconstruction, and return xarray data.
           ...

``accepted_options`` lists every keyword that ``reconstruct`` reads;
:meth:`hypermri.loading.experiment.MRIExperiment.reconstruct` rejects any other
option with a ``ValueError`` before data are read.  Import the class in
``src/hypermri/reconstruction/selection.py`` and add one entry per lower-case
``(vendor, sequence, reco_type)`` key to ``RECONSTRUCTIONS``, with its default
options:

.. code-block:: python

   ("bruker", "my_sequence", "raw"): (BrukerMySequenceRecoStrategy, _FLOAT32),
   ("bruker", "my_sequence", "seq2d"): (Brukerseq2dRecoStrategy, _NONE),

``raw`` is used for reconstruction from ``fid`` or ``ser`` data. ``seq2d`` is
used for scanner reconstructed Bruker ``2dseq`` data.  Each sequence key must
be ``default`` or a name that ``sequence_mapping.yaml`` resolves, and every
default option must appear in the class's ``accepted_options``; the tests in
``tests/unit/reconstruction/test_strategy_type_mapping.py`` check both.
A sequence entry whose method patterns lack an ``exact:``, ``contains:`` or
``startswith:`` prefix, or whose ``exact:`` alias already identifies another
sequence, raises ``ConfigurationError``.

The option ``data_precision`` accepts ``"float16"``, ``"float32"`` or
``"float64"``.  A strategy reads it with
:func:`hypermri.reconstruction.precision.parse_data_precision`, which raises
``ValueError`` for any other value, converts the acquired FID with
:func:`~hypermri.reconstruction.precision.convert_for_calculation` and the returned
arrays with :func:`~hypermri.reconstruction.precision.convert_for_storage`.  Read the
option before any data, so that a wrong value stops the reconstruction at
once.

Adding a New Vendor
-------------------

Vendors are detected through :mod:`hypermri.vendors.registry`. A vendor handler
is responsible for recognizing scan directories, loading raw/scanner data, and
exposing normalized parameters.

Typical steps:

1. Add a handler module under ``src/hypermri/vendors/``.
2. Implement an ``is_valid_scan_dir(path)`` check.
3. Implement metadata/data loading methods used by ``MRIExperiment`` and
   reconstruction strategies, such as ``get_parameters()``, raw-data access,
   and scanner-reconstruction access where applicable.
4. Register the vendor in the vendor registry configuration.
5. Add reconstruction strategies in ``reconstruction/strategies/<vendor>.py``
   or a vendor package.
6. Add entries for the vendor to ``RECONSTRUCTIONS`` in
   ``reconstruction/selection.py``.

At minimum, the vendor name used by the handler must match the vendor in those
keys because reconstruction lookup is keyed as
``(vendor, sequence, reco_type)``.

Adding a New Parameter
----------------------

Parameters such as ``n_slices`` should be normalized by the vendor handler.
``BaseSequence`` stores them in the sequence's ``params`` dictionary, the one
parameter store of an experiment; a parameter of the canonical schema can also
be read as an attribute, such as ``self.n_slices``.

The preferred flow is:

1. Add extraction logic in the relevant vendor handler so
   ``handler.get_parameters()`` returns the new key.
2. If the parameter belongs in the standardized parameter model, add it there
   as well. Otherwise, it is still stored because ``BaseSequence`` passes
   ``allow_extra=True``.
3. Use the parameter from sequence code as ``self.params["n_slices"]`` or,
   for a parameter of the canonical schema, as ``self.n_slices``.  Read a
   non-canonical parameter with ``self.get("name")``.
4. Add focused tests for both the handler parameter extraction and the sequence
   wrapper behavior.

Avoid deriving the same parameter separately in each sequence class unless the
meaning is sequence-specific. Shared scanner metadata belongs in the vendor
handler so all sequence classes see one consistent value.

Changing Automatic Load Behavior
--------------------------------

Most sequence classes should inherit the defaults from ``BaseAnatomy`` or
``BaseSpectroscopy``. If a sequence needs different behavior, set
``RECONSTRUCTION_POLICY`` on the class:

.. code-block:: python

   class MySequence(BaseAnatomy):
       RECONSTRUCTION_POLICY = {
           "lazy": (),
           "minimal": ("seq2d",),
           "full": ("seq2d", "raw"),
       }

Do not put automatic reconstruction calls in sequence ``__init__`` methods.
That makes loading harder to reason about and can re-read ``2dseq`` data that
was already loaded.
