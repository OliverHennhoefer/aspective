"""Ensembles of trees and boosting variants."""

from ._dart import DARTClassifier, DARTRegressor
from ._llf import LocalLinearForestRegressor
from ._logitboost import LogitBoostClassifier
from ._multiboost import MultiBoostClassifier
from ._rotation_forest import RotationForestClassifier

__all__ = [
    "DARTClassifier",
    "DARTRegressor",
    "LocalLinearForestRegressor",
    "LogitBoostClassifier",
    "MultiBoostClassifier",
    "RotationForestClassifier",
]
