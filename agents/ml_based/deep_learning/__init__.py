"""
Deep-learning ML agents.
"""

from importlib import import_module

_DEEP_AGENT_SPECS = {
    "CNNAgent": ("agents.ml_based.deep_learning.cnn_agent", "CNNAgent"),
    "DenseNNAgent": ("agents.ml_based.deep_learning.nn_classification_agent", "DenseNNAgent"),
    "LSTMAgent": ("agents.ml_based.deep_learning.lstm_agent", "LSTMAgent"),
    "TransformerAgent": ("agents.ml_based.deep_learning.transformer_agent", "TransformerAgent"),
}


def _load_deep_agents():
    deep_agents = {}
    unavailable_agents = {}

    for name, (module_name, attr_name) in _DEEP_AGENT_SPECS.items():
        try:
            module = import_module(module_name)
            deep_agents[name] = getattr(module, attr_name)
        except ModuleNotFoundError as exc:
            unavailable_agents[name] = exc

    return deep_agents, unavailable_agents


DEEP_LEARNING_AGENTS, UNAVAILABLE_DEEP_LEARNING_AGENTS = _load_deep_agents()
globals().update(DEEP_LEARNING_AGENTS)

__all__ = tuple(DEEP_LEARNING_AGENTS.keys()) + (
    "DEEP_LEARNING_AGENTS",
    "UNAVAILABLE_DEEP_LEARNING_AGENTS",
)
