import pytest
import torch

from witwin.maxwell.compiler import CompiledLumpedElement, compile_lumped_elements
from witwin.maxwell.lumped import Capacitor, Inductor, Resistor
from witwin.maxwell.scene import BoundarySpec, Domain, GridSpec, Scene, prepare_scene


def _scene(*, lumped_elements=()):
    return Scene(
        domain=Domain(((0.0, 1.0), (0.0, 1.0), (0.0, 1.0))),
        grid=GridSpec.uniform(0.25),
        boundary=BoundarySpec.none(),
        lumped_elements=lumped_elements,
        device="cpu",
    )


def test_scene_add_clone_and_prepare_preserve_lumped_elements():
    resistor = Resistor(
        "r1", positive=(0.75, 0.5, 0.5), negative=(0.25, 0.5, 0.5), resistance=50.0
    )
    capacitor = Capacitor(
        "c1",
        positive=(0.5, 0.75, 0.5),
        negative=(0.5, 0.25, 0.5),
        capacitance=2.0e-12,
    )
    scene = _scene(lumped_elements=(resistor,))

    assert scene.add_lumped_element(capacitor) is scene
    assert scene.lumped_elements == [resistor, capacitor]
    assert tuple(item.element_name for item in scene.compile_lumped_elements()) == (
        "r1",
        "c1",
    )

    cloned = scene.clone()
    prepared = prepare_scene(scene)
    assert cloned.lumped_elements == [resistor, capacitor]
    assert cloned.lumped_elements is not scene.lumped_elements
    assert prepared.lumped_elements == [resistor, capacitor]


@pytest.mark.parametrize("invalid", [object(), "resistor"])
def test_scene_rejects_unsupported_lumped_element_types(invalid):
    with pytest.raises(TypeError, match="Resistor, Capacitor, or Inductor"):
        _scene(lumped_elements=(invalid,))


def test_scene_rejects_duplicate_lumped_element_names():
    first = Resistor("load", (0.75, 0.5, 0.5), (0.25, 0.5, 0.5), 50.0)
    second = Capacitor("load", (0.5, 0.75, 0.5), (0.5, 0.25, 0.5), 1.0e-12)

    with pytest.raises(ValueError, match="Lumped element name 'load'.*already present"):
        _scene(lumped_elements=(first, second))


def test_scene_rejects_diagonal_lumped_element_terminals():
    diagonal = Inductor("diagonal", (0.75, 0.75, 0.5), (0.25, 0.25, 0.5), 1.0e-9)

    with pytest.raises(ValueError, match="'diagonal'.*axis-aligned"):
        _scene(lumped_elements=(diagonal,))


@pytest.mark.parametrize(
    ("positive", "negative", "terminal", "axis"),
    [
        ((1.25, 0.5, 0.5), (0.25, 0.5, 0.5), "positive", "x"),
        ((0.75, 0.5, 0.5), (0.25, -0.25, 0.5), "negative", "y"),
    ],
)
def test_scene_rejects_lumped_element_terminals_outside_domain(
    positive,
    negative,
    terminal,
    axis,
):
    element = Resistor("outside", positive, negative, 50.0)

    with pytest.raises(
        ValueError,
        match=rf"'outside'.*{terminal} terminal.*{axis}=.*outside.*Scene domain",
    ):
        _scene(lumped_elements=(element,))


def test_compile_lumped_elements_reuses_sparse_voltage_geometry_on_target_device():
    resistance = torch.tensor(50.0, dtype=torch.float64, requires_grad=True)
    elements = (
        Resistor("rx", (0.75, 0.5, 0.5), (0.25, 0.5, 0.5), resistance),
        Capacitor("cy", (0.5, 0.25, 0.5), (0.5, 0.75, 0.5), 2.0e-12),
        Inductor("lz", (0.5, 0.5, 0.75), (0.5, 0.5, 0.25), 3.0e-9),
    )
    prepared = prepare_scene(_scene(lumped_elements=elements))

    compiled = prepared.compile_lumped_elements(device=torch.device("cpu"))
    direct = compile_lumped_elements(prepared, device="cpu")

    assert [item.element_name for item in direct] == ["rx", "cy", "lz"]
    assert [item.element_name for item in compiled] == ["rx", "cy", "lz"]
    assert [item.kind for item in compiled] == ["resistor", "capacitor", "inductor"]
    assert [item.axis for item in compiled] == ["x", "y", "z"]
    assert [item.direction for item in compiled] == [1, -1, 1]
    assert [item.voltage_component for item in compiled] == ["Ex", "Ey", "Ez"]

    for item in compiled:
        assert isinstance(item, CompiledLumpedElement)
        assert item.port_name == item.element_name
        assert item.voltage_indices.device.type == "cpu"
        assert item.voltage_indices.dtype == torch.int64
        assert item.voltage_weights.device.type == "cpu"
        assert item.voltage_weights.dtype == torch.float64
        assert item.voltage_indices.shape == (2, 3)
        torch.testing.assert_close(
            torch.abs(item.voltage_weights),
            torch.full((2,), 0.25, dtype=torch.float64),
        )

    assert compiled[0].value is resistance
    assert compiled[0].value.requires_grad
    torch.testing.assert_close(
        compiled[0].integrate_voltage(
            {"Ex": torch.ones((5, 5, 5), dtype=torch.float64)}
        ),
        torch.tensor(0.5, dtype=torch.float64),
    )
    torch.testing.assert_close(
        compiled[1].integrate_voltage(
            {"Ey": torch.ones((5, 5, 5), dtype=torch.float64)}
        ),
        torch.tensor(-0.5, dtype=torch.float64),
    )


def test_compile_lumped_elements_reports_off_grid_terminals():
    element = Resistor("off_grid", (0.7, 0.5, 0.5), (0.25, 0.5, 0.5), 50.0)
    prepared = prepare_scene(_scene(lumped_elements=(element,)))

    with pytest.raises(ValueError, match=r"x node=0.7 must lie on the Yee x node grid"):
        prepared.compile_lumped_elements()
