Physics
========

MRI physics calculations and constants.

Thermal polarization reference
------------------------------

``get_thermal_polarisation_ratio`` returns the upper-to-lower Zeeman population
ratio. ``get_thermal_polarisation_ppm`` returns the population difference
normalized to all spins, expressed in ppm. For the retained spin-half model:

.. math::

   r = \frac{N_{\mathrm{upper}}}{N_{\mathrm{lower}}}
     = \exp\left(-\frac{\Delta E}{k_{\mathrm B}T}\right),
   \qquad
   P = \frac{1-r}{1+r}
     = \tanh\left(\frac{\Delta E}{2k_{\mathrm B}T}\right).

The ppm function returns :math:`10^6P`. Temperatures may be supplied in Kelvin
or Celsius; magnetic field strength is in Tesla. Import it from
``hypermri.utils.physics``; the former ``hypermri.utils.calculations`` module
was removed on 15 November 2026.
The normalization follows `Tateishi et al., Equation 1
<https://pmc.ncbi.nlm.nih.gov/articles/PMC4040576/>`_.

The owner approved correcting the existing ppm function on 10 October 2026.
For carbon-13 at 7 T and 230 K, it now returns 7.820554 ppm instead of
15.640986 ppm. The previous expression omitted the denominator :math:`1+r`.
Recompute analyses that used this helper; stored results and other signal
corrections are not automatically rescaled.

Polarization workflows retained for future use
----------------------------------------------

The thermal reference and existing RF/relaxation correction helpers remain
available for future polarization work. No current experimental determination
caller was identified. The repository does not provide a complete public
workflow for deriving absolute experimental polarization from a hyperpolarized
measurement and a thermal reference. ``PolarizationMonitoring13C`` represents
the acquisition sequence; its name does not imply such a calculation.

A future determination workflow needs an explicit contract for integrated signal
regions, receiver gain and averaging, spin amount, flip angles, acquisition
timing and relaxation corrections. Those choices are not inferred by the thermal
helper.

RF depletion and relaxation share
``hypermri.analysis.polarization.simulate_hyperpolarized_pulse_train``.
It accepts flip angles in radians and an explicit longitudinal retention factor
between pulses. Existing degree-based RF/T1 helpers retain their signatures,
angle interpolation and return arrays. This model describes hyperpolarized
depletion without equilibrium recovery; it does not determine an experimental
absolute polarization.

.. autofunction:: hypermri.analysis.polarization.simulate_hyperpolarized_pulse_train

.. automodule:: hypermri.utils.physics
   :members:
   :undoc-members:
   :show-inheritance:
