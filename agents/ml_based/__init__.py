"""
Curated exports for the currently stable `agents.ml_based` surface.

Only agents in the "keep" bucket are re-exported here. Experimental and legacy
files remain importable by direct path, but are intentionally excluded from the
default package surface until they are refactored.
"""

from importlib import import_module

_KEEP_AGENT_SPECS = {
    "AutoencoderAgent": ("agents.ml_based.autoencoder_agent", "AutoencoderAgent"),
    "ClusteringFilteredKNNAgent": ("agents.ml_based.clustering_agent", "ClusteringFilteredKNNAgent"),
    "CNNAgent": ("agents.ml_based.cnn_agent", "CNNAgent"),
    "HMMRegimeAgent": ("agents.ml_based.hmm_based_agent", "HMMRegimeAgent"),
    "KNNAgent": ("agents.ml_based.knn_agent", "KNNAgent"),
    "LRAgent": ("agents.ml_based.logistic_reg_agent", "LRAgent"),
    "LSTMAgent": ("agents.ml_based.lstm_agent", "LSTMAgent"),
    "NaiveBayesAgent": ("agents.ml_based.naive_bayes_agent", "NaiveBayesAgent"),
    "DenseNNAgent": ("agents.ml_based.nn_classification_agent", "DenseNNAgent"),
    "SVMAgent": ("agents.ml_based.svm_agent", "SVMAgent"),
    "TransformerAgent": ("agents.ml_based.transformer_agent", "TransformerAgent"),
}


def _load_keep_agents():
    keep_agents = {}
    unavailable_agents = {}

    for name, (module_name, attr_name) in _KEEP_AGENT_SPECS.items():
        try:
            module = import_module(module_name)
            keep_agents[name] = getattr(module, attr_name)
        except ModuleNotFoundError as exc:
            unavailable_agents[name] = exc

    return keep_agents, unavailable_agents


KEEP_AGENTS, UNAVAILABLE_KEEP_AGENTS = _load_keep_agents()
globals().update(KEEP_AGENTS)

REFACTOR_AGENTS = ()

ARCHIVE_CANDIDATES = (
    "BNNclassification",
    "DeepQLearningAgent",
    "LSTMAnomalyAgent",
    "LSTMAttentionAgent",
    "NNClassificationAgg",
    "TCNAgent",
    "TransformerTradingAgent",
    "VAEAgent",
)

__all__ = tuple(KEEP_AGENTS.keys()) + (
    "KEEP_AGENTS",
    "UNAVAILABLE_KEEP_AGENTS",
    "REFACTOR_AGENTS",
    "ARCHIVE_CANDIDATES",
)
