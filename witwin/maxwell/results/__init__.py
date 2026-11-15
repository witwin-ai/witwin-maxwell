"""Dimension-labelled, torch-native result data types."""

from .tensor_data import TensorData
from .typed import FarFieldData, FieldData, FluxData, ModeAmplitudeData, SParameterData, field_component_units

__all__ = [
    "TensorData",
    "FieldData",
    "FluxData",
    "ModeAmplitudeData",
    "SParameterData",
    "FarFieldData",
    "field_component_units",
]
