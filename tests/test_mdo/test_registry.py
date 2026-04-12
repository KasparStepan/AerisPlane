import pytest
from aerisplane.mdo.registry import DisciplineRegistry, DISCIPLINE_ORDER


def test_discipline_order_contains_all():
    expected = ["weights", "aero", "structures", "stability", "control", "propulsion", "mission"]
    assert DISCIPLINE_ORDER == expected


def test_registry_has_default_disciplines():
    reg = DisciplineRegistry()
    for name in DISCIPLINE_ORDER:
        assert reg.has(name), f"Missing discipline: {name}"


def test_register_custom_discipline():
    reg = DisciplineRegistry()

    def my_runner(aircraft, condition, results, **kwargs):
        return {"my_value": 42.0}

    reg.register("my_disc", my_runner, after="aero")
    assert reg.has("my_disc")


def test_register_after_sets_order():
    reg = DisciplineRegistry()

    def my_runner(aircraft, condition, results, **kwargs):
        return {"my_value": 42.0}

    reg.register("my_disc", my_runner, after="stability")
    order = reg.ordered_names()
    assert order.index("my_disc") > order.index("stability")


def test_register_duplicate_raises():
    reg = DisciplineRegistry()

    def my_runner(aircraft, condition, results, **kwargs):
        return {}

    with pytest.raises(ValueError, match="already registered"):
        reg.register("aero", my_runner)


def test_run_chain_subset():
    """run_chain with disciplines=['weights'] only calls weights runner."""
    reg = DisciplineRegistry()
    called = []

    def fake_weights(aircraft, condition, results, **kwargs):
        called.append("weights")
        return object()

    reg._runners["weights"] = fake_weights

    aircraft = object()
    condition = object()
    result = reg.run_chain(
        disciplines=["weights"],
        aircraft=aircraft,
        condition=condition,
    )
    assert called == ["weights"]
    assert "weights" in result
    assert "aero" not in result
