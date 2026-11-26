import hypermri.utils.physics as physics


def test_former_physics_noise_estimator_is_removed():
    """``hypermri.analysis.spectroscopy.estimate_noise`` is the only estimator."""
    assert not hasattr(physics, "estimate_noise")
