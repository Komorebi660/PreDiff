"""Model registry."""

from models.prediff import PreDiff

MODEL_REGISTRY = {
    'prediff': PreDiff,
}


def get_model_class(name):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name]
