"""
Classical sklearn-style ML agents.
"""

from agents.ml_based.classical.gaussian_process_agent import GaussianProcessAgent
from agents.ml_based.classical.knn_agent import KNNAgent
from agents.ml_based.classical.logistic_reg_agent import LRAgent
from agents.ml_based.classical.naive_bayes_agent import NaiveBayesAgent
from agents.ml_based.classical.online_sgd_agent import OnlineSGDAgent
from agents.ml_based.classical.quantile_regression_agent import QuantileRegressionAgent
from agents.ml_based.classical.qda_agent import QDAAgent
from agents.ml_based.classical.spline_logistic_agent import SplineLogisticAgent
from agents.ml_based.classical.svm_agent import SVMAgent

__all__ = ("GaussianProcessAgent", "KNNAgent", "LRAgent", "NaiveBayesAgent", "OnlineSGDAgent", "QuantileRegressionAgent", "QDAAgent", "SplineLogisticAgent", "SVMAgent")
