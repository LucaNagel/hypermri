Registry Overview
=================

.. note::

   Registry extension APIs and their YAML schemas are **Beta**. Registry
   presence describes implementation availability, not stable support. The
   canonical support contract is :doc:`../support`.

HyperMRI resolves runtime behaviour through registries with Python extension
APIs and validated external settings. Vendor and sequence types have explicit
Python mappings; reconstruction uses one Python table without a registry.
Packaged YAML owns external detection patterns, scanner aliases and categories.
Existing custom dotted class paths retain their validated compatibility readers.

Vendor registry
---------------

Implemented in :mod:`hypermri.vendors.registry`. The registry resolves canonical
vendor identifiers to explicit Python handler types. The packaged
``vendor_mapping.yaml`` contains only detection patterns and their priorities;
its order, patterns and priorities are unchanged. Detection combines handler
``is_valid_scan_dir`` hooks with filename/pattern checks.
:func:`hypermri.create_experiment` performs detection automatically. Advanced
users can register a Python handler type through
:func:`hypermri.vendors.register_vendor`.

Custom external settings may only use maintained vendor identifiers; the file
names no Python classes, and a ``handler_class`` field is rejected with a
pointer to :func:`~hypermri.vendors.register_vendor`. Missing files, unknown
identifiers, repeated keys and unknown/missing fields raise actionable
configuration errors.
Handler construction, scan loading and acquired-data precision are unchanged.

Sequence registry
-----------------

Backed by :mod:`hypermri.sequences.registry`. The 32 maintained canonical IDs
resolve through one explicit Python type mapping.
``sequences/config/sequence_mapping.yaml`` owns categories, raw method patterns
and optional vendor scopes. The registry powers ``Study`` and
:class:`hypermri.loading.experiment.MRIExperiment` when they instantiate sequence
classes; reconstruction alias resolution uses the same settings.

Custom files may only use maintained IDs; the file names no Python classes, and
a ``class`` field is rejected with a pointer to
:func:`~hypermri.sequences.register_sequence`. Unknown IDs fail with the
file/key and a remedy. Persistent scanner aliases belong in YAML; direct Python types can be registered
with :func:`hypermri.sequences.register_sequence`. Exact aliases are searched
before broader patterns, with vendor scopes and file order preserved. Registry
construction imports definitions without constructing sequences or reading data.

Reconstruction table
--------------------

:data:`hypermri.reconstruction.selection.RECONSTRUCTIONS` maps 73
vendor/sequence/reconstruction keys to 13 maintained strategy classes and their
read-only default options; there is no YAML file and no runtime registration.
During :meth:`hypermri.loading.experiment.MRIExperiment.reconstruct`,
:func:`~hypermri.reconstruction.selection.find_reconstruction` selects the
exact method entry, then its canonical sequence, then the vendor default, and
returns the class with a copy of its defaults. ``reconstruct`` rejects an option
that the class does not list in ``accepted_options`` before any data are read,
then builds a new strategy instance for the call. To add a reconstruction, add
an entry to the table and list the strategy's options in ``accepted_options``.

Direct visualization
--------------------

Plotting has no registry, factory, adapter or YAML mapping. Both sequence
plotting and fit review call :func:`hypermri.visualization.plot` with PlotOptions.
The same Plotly viewer owns state and rendering. Extend a returned controller
through ``register_plugin``; see :doc:`../api/public` for removed import paths.

Runtime overrides
-----------------

The vendor and sequence registries derive from
:class:`hypermri.loading.protocols.BaseRegistry`. They
share helpers to list entries and register runtime classes. Each registry reads
its settings file once during construction and validates it with
:mod:`hypermri.config.mappings`. A missing file, repeated key, unknown/missing
field, class field or unknown vendor or sequence identifier raises
:class:`~hypermri.exceptions.ConfigurationError` naming the file and entry.
No fallback copy replaces external settings. Their Python mappings supply code
objects; their YAML files own external detection and scanner alias settings.
The sequence mapping file is read, validated and matched against raw method
names by :mod:`hypermri.sequence_mapping`, which imports no sequence class, so
reconstruction can convert a method name to its canonical sequence ID without
importing the ``sequences`` package.

A reconstruction strategy is looked up by the exact sequence key, then by the canonical ID that :func:`hypermri.sequence_mapping.method_name_to_sequence_id` converts the raw method name to, and finally by the vendor's ``default`` entry. A sequence without its own entry therefore uses the vendor default; it never borrows the strategy of a sequence whose ID it merely contains.

Loading and error behavior
--------------------------

``load_scan`` returns a generic ``MRIExperiment`` for an unregistered method
and logs a warning with the method and vendor. Pass ``resolve_sequence=False``
to request that generic representation intentionally, or ``sequence_class``
to choose the representation explicitly. Registry import errors propagate.
The vendor-specific sequence mapping also owns the categories used by generic
reconstruction; method-name and class-name
substring guesses are no longer used. Unregistered methods retain the handler's
spectroscopy flag or the anatomical fallback.

Vendor detection skips unreadable folders (``OSError``). Other failures of a
handler's ``is_valid_scan_dir`` raise ``VendorError`` naming the handler and
chaining the original exception. Runtime patterns use the same validation as
the packaged mapping: every pattern needs a ``file:``, ``contains:``,
``extension:`` or ``dir:`` prefix. Unprefixed substring patterns were removed
on 15 November 2026 and raise ``ValueError``.

Viewer construction errors propagate. Unknown display options fail before
widgets, and unsupported backends fail explicitly rather than selecting another
plotting mode.
