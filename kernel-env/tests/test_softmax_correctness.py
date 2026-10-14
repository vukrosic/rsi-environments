import numpy as np
import pytest
import kernel_env.environment as envmod

@pytest.mark.parametrize("kind", ("missing_mass", "zero_mass", "negative_probability"))
def test_softmax_invariants_reject_small_elementwise_but_wrong_probabilities(kind):
    arrays = (np.zeros((2, 1024), dtype=np.float32),)
    expected = np.full((2, 1024), 1 / 1024, dtype=np.float32)
    actual = expected.copy()
    if kind == "missing_mass":
        actual *= 0.99
    elif kind == "zero_mass":
        actual.fill(0)
    else:
        actual[:, 0] = -0.0001
    assert np.abs(actual - expected).max() < 2e-3
    metadata = {"device": "cpu", "input_hashes": [envmod.fingerprint(arrays[0])]}
    with pytest.raises(envmod.InvalidCandidate, match="softmax"):
        envmod._validate(actual, expected, metadata, arrays, "cpu", 2e-3, task="softmax")
