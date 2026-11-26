Physical spatial filtering
==========================

Gaussian and Fermi windows share one numerical implementation for labelled
anatomical, CSI and other Read/Phase datasets. Widths use cycles/mm (1/mm).
Image data are Fourier transformed along Read and Phase only; k-space data
are weighted directly. Slice, echo, channel, repetition and spectral dimensions
remain independent. Coordinates, orientation, signal units and existing
attributes are retained; a JSON ``spatial_filter`` attribute records the options
and calibration. Filtering is eager and never changes the source array.

.. code-block:: python

    from hypermri.analysis.spatial_filtering import (
        SpatialFilterOptions,
        apply_spatial_filter,
    )

    gaussian = SpatialFilterOptions(sigma_per_mm=0.1)
    filtered = apply_spatial_filter(image, gaussian)
    fermi = SpatialFilterOptions(
        kind="fermi", radius_per_mm=0.15, transition_per_mm=0.02,
    )
    filtered_kspace = apply_spatial_filter(kspace, fermi)

Image coordinates must provide uniform separable mm/cm/m sampling (or
``resolution_mm``). For arrays with only indices, pass
``spatial_spacing_mm=(read_spacing_mm, phase_spacing_mm)`` explicitly. K-space
coordinates in reciprocal units may be nonuniform or descending. Legacy CSI
k-space arrays with spatial mm coordinates also need explicit image sampling;
those coordinates must not be mistaken for reciprocal distances. Declare
``domain="spatial"`` or ``domain="kspace"`` when the array's domain is absent or
ambiguous; the standalone function defaults to spatial only when absent.

At radius 0.1/mm with Gaussian sigma 0.1/mm, the weight is
``exp(-0.5)`` = 0.6065306597, regardless of matrix/FOV. Fermi uses the logistic
``1/(1+exp((r-radius)/transition))``, without normalization at zero; its weight
at the specified radius is 0.5 and a broad transition can attenuate DC.
Distances are cycles/mm, not radians/mm. These physical widths intentionally
replace the old preview's dimensionless fraction of acquired radius.

``spatial_to_kspace(image)`` returns the centred spatial k-space along Read and
Phase with the same forward transform, so a window drawn over its 1/mm axes
lies where ``apply_spatial_filter`` applies it. The browser view's "k-space"
controls show this copy.

CSI's raw adapter uses verified method, paired encoding positions and FOV/matrix
metadata. Fourier-encoded Luca methods weight acquired points directly;
Bruker CSI/csi_fix retain their established phase-modulated scanner-grid
convention and require a complete unique grid. Unknown raw methods/layouts
raise with guidance to supply a calibrated labelled dataset. This filter
does not add reconstruction support for deferred B items.

.. automodule:: hypermri.analysis.spatial_filtering
   :members:
   :undoc-members:
