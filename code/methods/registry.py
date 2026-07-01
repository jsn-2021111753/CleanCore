"""Method registry for the unified experiment runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from common.interfaces import MethodOutput
from methods.base import MethodContext
from methods import (
    cleanlab,
    cleancore,
    coretab,
    coretab_misdetect,
    ctrl,
    deem,
    deepfool,
    goodcore,
    gradmatch,
    herding,
    misdetect,
    misdetect_coretab,
)


@dataclass(frozen=True)
class RegisteredMethod:
    name: str
    run: Callable[[MethodContext], MethodOutput]
    category: str
    paper: str


METHOD_REGISTRY: Dict[str, RegisteredMethod] = {
    "cleanlab": RegisteredMethod("cleanlab", cleanlab.run, "data_cleaning", "Confident Learning: Estimating Uncertainty in Dataset Labels"),
    "misdetect": RegisteredMethod("misdetect", misdetect.run, "data_cleaning", "MisDetect: Iterative Mislabel Detection using Early Loss"),
    "ctrl": RegisteredMethod("ctrl", ctrl.run, "data_cleaning", "CTRL: Clustering Training Losses for Label Error Detection"),
    "herding": RegisteredMethod("herding", herding.run, "subset_selection", "Herding Dynamical Weights to Learn"),
    "deepfool": RegisteredMethod("deepfool", deepfool.run, "subset_selection", "DeepFool: A Simple and Accurate Method to Fool Deep Neural Networks"),
    "gradmatch": RegisteredMethod("gradmatch", gradmatch.run, "subset_selection", "GradMatch: Gradient Matching Based Data Subset Selection for Efficient Deep Model Training"),
    "coretab": RegisteredMethod("coretab", coretab.run, "subset_selection", "Datamap-Driven Tabular Coreset Selection for Classifier Training"),
    "misdetect_coretab": RegisteredMethod("misdetect_coretab", misdetect_coretab.run, "pipeline", "MisDetect followed by CoreTab"),
    "coretab_misdetect": RegisteredMethod("coretab_misdetect", coretab_misdetect.run, "pipeline", "CoreTab followed by MisDetect"),
    "goodcore": RegisteredMethod("goodcore", goodcore.run, "hybrid", "GoodCore: Data-effective and Data-efficient Machine Learning through Coreset Selection over Incomplete Data"),
    "deem": RegisteredMethod("deem", deem.run, "hybrid", "Two Birds with One Stone: Efficient Deep Learning over Mislabeled Data through Subset Selection"),
    "cleancore": RegisteredMethod("cleancore", cleancore.run, "proposed", "CleanCore Sections 3-4"),
}


def get_method(name: str) -> RegisteredMethod:
    key = str(name).lower()
    if key not in METHOD_REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Expected one of: {', '.join(sorted(METHOD_REGISTRY))}")
    return METHOD_REGISTRY[key]
