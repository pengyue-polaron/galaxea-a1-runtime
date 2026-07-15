import pytest

from galaxea_a1_runtime.configuration.base import float_tuple, floating


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_floating_rejects_non_finite_values(value):
    with pytest.raises(ValueError):
        floating({"value": value}, "value")


def test_float_tuple_rejects_non_finite_values():
    with pytest.raises(ValueError):
        float_tuple({"values": [0.0, float("nan")]}, "values", 2)
