"""
Classical sklearn-style ML agents.
"""

from agents.ml_based.classical.knn_agent import KNNAgent
from agents.ml_based.classical.logistic_reg_agent import LRAgent
from agents.ml_based.classical.naive_bayes_agent import NaiveBayesAgent
from agents.ml_based.classical.svm_agent import SVMAgent

__all__ = ("KNNAgent", "LRAgent", "NaiveBayesAgent", "SVMAgent")
