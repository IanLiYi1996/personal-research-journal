"""Offline algebra and sampling checks, not a DetectorScope benchmark.

No models, datasets, API clients, or detectorscope package are imported.
The sampling check loads only three reviewed, pure sampling methods and two
list helpers from the pinned Apache-2.0 source. It never constructs the full
feature validator or enters its semantic-negative/API paths.
"""

import ast
import itertools
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Any


def sigmoid(value):
    return 1.0 / (1.0 + math.exp(-value))


def check_linear():
    vector = [2.0, 1.0, 3.0]
    weights = [0.4, -0.3, 0.2]
    bias = -0.5
    contributions = [x * w for x, w in zip(vector, weights)]
    logit = bias + sum(contributions)
    ablations = []
    for index in range(len(vector)):
        changed = vector.copy()
        changed[index] = 0.0
        changed_logit = bias + sum(x * w for x, w in zip(changed, weights))
        ablations.append(logit - changed_logit)
    assert all(math.isclose(a, b) for a, b in zip(contributions, ablations))
    assert math.isclose(logit, bias + sum(contributions))
    probability_deltas = [
        sigmoid(logit) - sigmoid(logit - delta) for delta in ablations
    ]
    assert not math.isclose(
        sigmoid(logit), sigmoid(bias) + sum(probability_deltas)
    )
    return {
        "toy_logit": logit,
        "toy_probability": sigmoid(logit),
        "contributions": contributions,
        "zero_ablation_deltas": ablations,
        "probability_is_not_additive": True,
    }


def check_nonlinear():
    # An AND-like toy tree, not a released XGBoost model.
    def score(vector):
        return 1.0 if vector[0] > 0.5 and vector[1] > 0.5 else 0.0

    original = score([1.0, 1.0])
    bias = score([0.0, 0.0])
    deltas = [original - score([0.0, 1.0]), original - score([1.0, 0.0])]
    residual = original - bias - sum(deltas)
    assert residual != 0.0
    return {"toy_zero_ablation_deltas": deltas, "toy_residual": residual}


def check_sampling():
    source = Path(__file__).resolve().parents[1] / "source"
    tree = ast.parse((source / "detectorscope/autointerpret.py").read_text())
    helpers = {"interleave_lists", "nested_shuffle"}
    methods = {"_equalize_quantiles", "_sample_examples", "_sample_groups"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in helpers
    ]
    validator = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "FeatureDescriptionValidator"
    )
    selected = [
        node
        for node in validator.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    assert len(nodes) == len(helpers) and len(selected) == len(methods)
    nodes.append(
        ast.ClassDef(
            name="SamplingOnly",
            bases=[],
            keywords=[],
            body=selected,
            decorator_list=[],
        )
    )
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {
        "Any": Any,
        "random": random,
        "itertools": itertools,
        "more_itertools": SimpleNamespace(
            chunked=lambda items, size: (
                items[start : start + size]
                for start in range(0, len(items), size)
            )
        ),
    }
    exec(compile(module, "<reviewed-pure-sampling-methods>", "exec"), namespace)
    sampler = namespace["SamplingOnly"]()
    sampler.prompt_sampling_method = "stratified"
    sampler.negative_sampling_method = "uniform"
    sampler.group_size = 5
    bins = {
        quantile: list(range(offset * 100, offset * 100 + 20))
        for offset, quantile in enumerate([-1, 0, 1, 2, 3])
    }
    random.seed(42)
    generation = sampler._sample_groups(bins, 10)
    random.seed(42)
    validation = sampler._sample_groups(bins, 50)
    generation_ids = {doc for group in generation for _, doc in group}
    validation_ids = {doc for group in validation for _, doc in group}
    overlap = generation_ids & validation_ids
    assert len(overlap) == 10
    return {
        "synthetic_generation_documents": len(generation_ids),
        "synthetic_validation_documents": len(validation_ids),
        "overlap": len(overlap),
        "scope": "synthetic bins; stratified sampling; uniform negatives; seed 42",
        "paper_experiment_reproduced": False,
    }


if __name__ == "__main__":
    print(
        json.dumps(
            {
                "linear_algebra": check_linear(),
                "nonlinear_counterexample": check_nonlinear(),
                "sampling_path": check_sampling(),
                "detector_inference_run": False,
                "paid_api_calls": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
