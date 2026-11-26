Configuration ownership
=======================

HyperMRI uses configuration for scanner and workflow mappings, but not every
constant should be configurable.  A value is easier to understand when its
location says who owns it and whether changing it is scientifically safe.

This page defines that ownership.  It is both a rule for new code and a guide
for simplifying existing configuration.

The central rule
----------------

Configuration files contain **data selected by a user or deployment**.  Python
modules contain **data-model contracts and algorithm definitions**.  Scientific
options live next to the algorithm that interprets them.

Do not add a catch-all ``defaults.yaml``.  HyperMRI previously shipped such a
file, but no code loaded it.  Its apparent defaults therefore differed from the
values used by reconstruction and visualization.  It was removed rather than
wired into unrelated subsystems.

Kinds of settings
-----------------

Use the following decision table when placing a value.

.. list-table::
   :header-rows: 1
   :widths: 22 27 30 21

   * - Kind
     - Examples
     - Owner
     - Representation
   * - Domain invariant
     - Canonical dimension names, required result variables, coordinate
       conventions
     - The domain package, for example ``analysis.spectroscopy``
     - Named Python constants and small validation functions
   * - Scientific option
     - Fit bounds, line broadening, peak distance, reconstruction precision
     - The function or workflow that interprets the value
     - A typed options object with documented units
   * - Scanner mapping
     - Vendor detection, raw method aliases, reconstruction selection
     - Loading and reconstruction
     - Validated YAML with one owning loader
   * - Support contract
     - Canonical sequences covered by stable end-to-end tests
     - Release and validation policy
     - Versioned JSON
   * - Presentation preference
     - Colormap, initial slice, overlay opacity
     - Visualization entry point or saved viewer state
     - Explicit arguments or a typed plotting-options object
   * - Operational setting
     - Log level, progress reporting, worker count
     - Application entry point
     - Explicit arguments, environment variables, or a runtime settings object

Domain invariants
-----------------

A domain invariant defines the shape or meaning of data.  It must not change
because a local configuration file happened to be edited.  Keep it close to
the normalization and validation code that enforces it.

For example, ``STANDARD_SPECTROSCOPY_DIMS`` is a spectroscopy data contract,
not a user preference.  It therefore lives together with dimension aliases and
normalization helpers in ``hypermri.analysis.spectroscopy.schema`` rather than
in YAML.

Each invariant should have:

* a name that states the scientific concept rather than the implementation;
* a short comment or docstring explaining meaning and units;
* validation at the public data boundary; and
* tests using both valid and invalid data.

Scientific options
------------------

An algorithm should receive its adjustable parameters explicitly.  Prefer a
small immutable dataclass when a workflow has more than a few related options.
The class belongs in the same module as the public algorithm, or in a nearby
``options.py`` when several algorithms share it.

For example, spectroscopy fitting may expose a
``SpectroscopyFitOptions`` object containing fit bounds, line broadening and
parallel-execution settings.  Names must include units where ambiguity is
possible, such as ``frequency_tolerance_hz`` rather than ``tolerance``.

For reproducibility:

* defaults are declared in exactly one place;
* the effective options are attached to or serializable with the result;
* notebooks can override options without editing package files;
* array-valued initial estimates remain labelled ``xarray`` data rather than
  being hidden in a global settings object; and
* changes to scientific defaults require numerical regression tests and a
  changelog entry.

Mapping files
-------------

The current package contains the following active mapping and policy files:

.. list-table::
   :header-rows: 1
   :widths: 35 30 35

   * - File
     - Owner
     - Purpose
   * - ``vendors/config/vendor_mapping.yaml``
     - ``hypermri.vendors.registry``
     - Vendor priorities/detection patterns; maintained types belong to Python
   * - ``sequences/config/sequence_mapping.yaml``
     - ``hypermri.sequences.registry``
     - Sequence categories/raw aliases/vendor scopes; maintained types belong to Python
   * - ``config/support_policy.json``
     - Release validation
     - Machine-readable stable raw-reconstruction allowlist

These paths describe the current implementation.

Each registry validates its own file when it is constructed, using the shared
strict reader in :mod:`hypermri.config.mappings`.  Loading stops with a
:class:`~hypermri.exceptions.ConfigurationError` naming the file and the
entry if

* the file is missing or not valid YAML, or a key is repeated within one
  mapping;
* a top-level section or an entry field is missing or unknown;
* a configured class cannot be imported, or a classless vendor/sequence/reconstruction key has
  no maintained Python type;
* a vendor priority is not an integer or a vendor detection pattern lacks one
  of the prefixes ``file:``, ``contains:``, ``extension:`` or ``dir:``;
* a sequence category is unknown, a method pattern lacks one of the prefixes
  ``exact:``, ``contains:`` or ``startswith:``, or a canonical ID or
  ``exact:`` alias identifies two sequences of the same vendor scope;
* a strategy entry names a sequence that the sequence mapping does not
  resolve, a reconstruction type other than ``raw`` or ``seq2d``, or no
  ``options`` mapping (write ``options: {}`` for none).

``contains:`` and ``startswith:`` patterns of different sequences may overlap,
because exact aliases are matched first and the remaining patterns in file
order.  The individual strategy option names are not yet validated against the
options each strategy accepts.

Every retained mapping file must satisfy all of these requirements:

1. It has one documented owning loader.
2. Its schema and accepted keys are validated before processing data.
3. Unknown keys and invalid import paths produce a clear error.
4. A canonical ID has one definition; hard-coded fallback copies are avoided.
5. Package-data metadata includes the file.
6. A test verifies that the installed package can load it.
7. Cross-file references, such as support-policy sequence IDs, are checked.

If the application can safely continue without an optional mapping entry, one
startup validation report may collect all invalid entries.  Silently omitting
individual entries is not an acceptable fallback because subsequent vendor or
sequence detection then becomes difficult to diagnose.

Where to add a new value
------------------------

Ask these questions in order:

1. Does changing the value change the meaning or required structure of data?
   If yes, it is a domain invariant in Python.
2. Does the algorithm need the value to reproduce a scientific result?  If
   yes, it is a typed option owned by that algorithm.
3. Does the value map an external scanner name to a HyperMRI concept?  If yes,
   it belongs in the mapping owned by loading or reconstruction.
4. Does it only affect the current display?  If yes, it is a plotting argument
   or saved viewer state.
5. Does it control the running process rather than scientific behavior?  If
   yes, it belongs to explicit runtime settings.

If none applies, prefer an ordinary named local variable.  Making a value
globally configurable increases the public surface and requires validation,
documentation and compatibility guarantees.

Migration rules
---------------

Configuration cleanup should be behavior-preserving and incremental:

* first identify the actual readers of each file and record existing behavior;
* then add schema and consistency tests;
* remove dead files rather than inventing new runtime behavior for them;
* consolidate duplicate mappings only while simplifying the owning workflow;
* introduce an explicit deprecation period for supported user overrides; and
* never mix a change to a scientific default with a file move or registry
  refactor.

This separation makes reviews easier: a structural pull request should produce
the same numerical data, while a scientific change should clearly show the
changed parameter and its validation evidence.
