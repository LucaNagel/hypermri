System
======

System interaction, logging, and environment helpers.

Logging
-------

``set_hypermri_log_level("warning")`` sets the shared ``hypermri`` threshold
for existing and subsequently imported modules. The package owns one fallback
stream handler, so initializing a ``Study`` or configuring the level repeatedly
does not duplicate messages. Module-specific fallback formats remain available.

Explicit child levels, including levels selected through vendor ``log_mode``
arguments, remain authoritative. Unlike the previous implementation, the package
setter does not overwrite them. To restore package inheritance for a child,
call ``logging.getLogger(child_name).setLevel(logging.NOTSET)``. Importing a
sequence module no longer imposes a separate info, warning or critical threshold.
``Study(log_mode=...)`` configures the shared threshold without pinning a
separate study-logger level.

Existing application handlers, their levels and formats, and propagation flags
are preserved. When an application handler lies on a record's propagation path,
the package fallback yields to it, including after later application logging
configuration. Application-installed handlers keep their own routing behavior.

.. automodule:: hypermri.utils.system.logging
   :members:
   :undoc-members:
   :show-inheritance:

Decorators
----------
.. automodule:: hypermri.utils.system.decorators
   :members:
   :undoc-members:
   :show-inheritance:

Environment
-----------
.. automodule:: hypermri.utils.system.environment
   :members:
   :undoc-members:
   :show-inheritance:

Objects
-------
.. automodule:: hypermri.utils.system.objects
   :members:
   :undoc-members:
   :show-inheritance:
