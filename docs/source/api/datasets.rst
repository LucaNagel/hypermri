Example datasets
================

Public example data are not part of the package. The package ships a small
registry of released datasets with their source, licence and checksum;
:func:`~hypermri.datasets.fetch_example_data` downloads one into a local
cache on first use. The command ``hypermri-download-example-data --list``
lists the released datasets from a shell.

.. autofunction:: hypermri.datasets.fetch_example_data

.. autofunction:: hypermri.datasets.list_datasets

.. autofunction:: hypermri.datasets.get_dataset_info

.. autofunction:: hypermri.datasets.is_dataset_cached

.. autoexception:: hypermri.datasets.DatasetError
