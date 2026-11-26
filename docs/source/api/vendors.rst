Vendor handlers
===============

Vendor implementations live under ``hypermri.vendors``. The root loading
API remains the normal entry for supported scans; direct handlers retain
their support/maturity boundaries from :doc:`../support`.

.. code-block:: python

   from hypermri.vendors.base import VendorHandler
   from hypermri.vendors.topspin import BrukerTopSpinHandler

The former ``hypermri.vendorhandler`` and ``hypermri.topspin_exp`` modules were
removed on 15 November 2026; pickles that name them no longer load. TopSpin
keeps ``hypermri.topspin_exp`` as its logging category name.

The legacy ``VendorFactory`` remains directly accessible in
``hypermri.vendors.base`` for its existing two-vendor dispatch only. New code
uses :func:`hypermri.load_scan` and its documented vendor/sequence detection.

Shared handler contract
-----------------------

.. autoclass:: hypermri.vendors.base.VendorHandler
   :members:

TopSpin spectroscopy
--------------------

.. autoclass:: hypermri.vendors.topspin.BrukerTopSpinHandler
   :members:
   :show-inheritance:

Construction reads acquisition metadata and prepares frequency axes, with
binary loading controlled by ``load_data``. The existing 70-sample cutoff,
binary interpretation, units and coordinates are unchanged by the relocation.
No new scientific validation or support level is implied by the import change.
