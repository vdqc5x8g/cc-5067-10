"""Identical-observation clipping teaching lab.

Public surface:
    Spec                 editable generator parameters (JSON round-trip)
    build_material       synthesize paired originals + observations
    blind_reconstruct    observation-only repair under two hypotheses
    threshold_lift_comparison   blind before/after comparison + exclusion proof
    build_lab_pack       write observation/truth/reference artifact tree
"""

from .spec import MaterialSpecError, Spec
from .material import LabMaterial, build_material
from .engine import blind_reconstruct, residual
from .evaluation import blind_comparison, exclusion_experiment, threshold_lift_comparison
from .build import build_lab_pack, lab_payload
from .methods import compare_construction_methods

__all__ = [
    "Spec",
    "MaterialSpecError",
    "LabMaterial",
    "build_material",
    "blind_reconstruct",
    "residual",
    "blind_comparison",
    "exclusion_experiment",
    "threshold_lift_comparison",
    "build_lab_pack",
    "lab_payload",
    "compare_construction_methods",
]
