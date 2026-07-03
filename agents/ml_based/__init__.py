"""
Curated exports for the currently stable `agents.ml_based` surface.

Files are grouped by model type under subpackages such as `classical`,
`deep_learning`, `anomaly`, `clustering`, `regime`, and
`reinforcement_learning`.
"""

from importlib import import_module

ML_TYPE_FOLDERS = (
    "anomaly",
    "classical",
    "clustering",
    "deep_learning",
    "regime",
    "reinforcement_learning",
)

ML_TYPE_MODULES = {
    "anomaly": (
        "autoencoder_agent",
        "lstm_anomaly_agent",
        "variational_encoder_agent",
    ),
    "classical": (
        "knn_agent",
        "logistic_reg_agent",
        "naive_bayes_agent",
        "svm_agent",
    ),
    "clustering": ("clustering_agent",),
    "deep_learning": (
        "bayesian_nn",
        "cnn_agent",
        "lstm_agent",
        "lstm_attention_agent",
        "nn_classification_agent",
        "nn_classification_aggregate",
        "nn_classsification_aggregate",
        "tcn_agent",
        "transformer_agent",
        "transformer_agent_test",
    ),
    "regime": ("hmm_based_agent",),
    "reinforcement_learning": (
        "deep_q_learning_agent",
        "dqn_test",
    ),
}

_KEEP_AGENT_SPECS = {
    "AutoencoderAgent": ("agents.ml_based.anomaly.autoencoder_agent", "AutoencoderAgent"),
    "ClusteringFilteredKNNAgent": ("agents.ml_based.clustering.clustering_agent", "ClusteringFilteredKNNAgent"),
    "CNNAgent": ("agents.ml_based.deep_learning.cnn_agent", "CNNAgent"),
    "HMMRegimeAgent": ("agents.ml_based.regime.hmm_based_agent", "HMMRegimeAgent"),
    "KNNAgent": ("agents.ml_based.classical.knn_agent", "KNNAgent"),
    "LRAgent": ("agents.ml_based.classical.logistic_reg_agent", "LRAgent"),
    "LSTMAgent": ("agents.ml_based.deep_learning.lstm_agent", "LSTMAgent"),
    "NaiveBayesAgent": ("agents.ml_based.classical.naive_bayes_agent", "NaiveBayesAgent"),
    "DenseNNAgent": ("agents.ml_based.deep_learning.nn_classification_agent", "DenseNNAgent"),
    "SVMAgent": ("agents.ml_based.classical.svm_agent", "SVMAgent"),
    "TransformerAgent": ("agents.ml_based.deep_learning.transformer_agent", "TransformerAgent"),
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

__all__ = tuple(KEEP_AGENTS.keys()) + (
    "ML_TYPE_FOLDERS",
    "ML_TYPE_MODULES",
    "KEEP_AGENTS",
    "UNAVAILABLE_KEEP_AGENTS",
    "REFACTOR_AGENTS",
)
