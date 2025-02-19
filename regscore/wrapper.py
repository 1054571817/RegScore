from typing import *

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from regscore.regscore import (RegScoreRegressor, RegScoreOptimizer)


class RegScore(BaseEstimator):
    """
    Wrapper for RegScore algorithm

    Parameters
    ----------
    k : int
        sparsity constraint, equivalent to number of selected (binarized) features in the final sparse model(s)
    select_top_m : int, optional
        number of top solutions to keep among the pool of diverse sparse solutions, by default 50
    gap_tolerance : float, optional
        tolerance in logistic loss for creating diverse sparse solutions, by default 0.05
    parent_size : int, optional
        how many solutions to retain after beam search, by default 10
    child_size : int, optional
        how many new solutions to expand for each existing solution, by default None
    featureIndex_to_groupIndex : ndarray, optional
        (1D array with `int` type) featureIndex_to_groupIndex[i] is the group index of feature i, by default None
    
    Attributes
    ----------
    beta0_ : List
        intercepts used in the final diverse sparse models
    betas_ : List
        coefficients used in the final diverse sparse models
    """

    def __init__(self, k: int = 10, select_top_m: int = 50, gap_tolerance: float = 0.05, parent_size: int = 10,
                 child_size: int = None,
                 featureIndex_to_groupIndex: np.ndarray = None) -> None:
        self.k = k
        self.select_top_m = select_top_m
        self.gap_tolerance = gap_tolerance
        self.parent_size = parent_size
        self.child_size = child_size
        self.featureIndex_to_groupIndex = featureIndex_to_groupIndex

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train RegScore

        Parameters
        ----------
        X : np.ndarray
            training data
        y : np.ndarray
            training data labels
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()

        opt = RegScoreOptimizer(
            X=X, y=y,
            k=self.k, select_top_m=self.select_top_m,
            gap_tolerance=self.gap_tolerance, parent_size=self.parent_size, child_size=self.child_size,
            featureIndex_to_groupIndex=self.featureIndex_to_groupIndex,
        )
        opt.optimize()  # train
        beta0, betas = opt.beta0, opt.betas
        self.beta0_, self.betas_ = opt.sparseLogRegModel_object.transform_coefficients_to_original_space(beta0, betas)
        self.is_fitted_ = True

        return self

    def predict(self, X: np.ndarray) -> np.ndarray[float]:
        """
        make bianry prediction

        Parameters
        ----------
        X : np.ndarray
            input data

        Returns
        -------
        np.ndarray[float]
            binary predictions
        """
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()

        assert hasattr(self, 'is_fitted_'), "Please fit the model first"

        y_pred = RegScoreRegressor(intercept=self.beta0_, coefficients=self.betas_).predict(X)

        return y_pred

    def get_model_params(self) -> Tuple[List[float], List[float]]:
        """
        Get model parameters for RegScore

        Returns
        -------
        beta0 (intercept), and betas (coefficients)
        """
        assert hasattr(self, 'is_fitted_'), "Please fit the model first"
        return self.beta0_, self.betas_

    def print_risk_card(self, feature_names: List[str], X_train: np.ndarray,
                        unit="point(s)") -> None:
        """
        print RegScore card

        Parameters
        ----------
        feature_names : list
            feature names for the features
        X_train : np.ndarray
        if provided, prints logistic loss on training set
        """
        assert hasattr(self, 'is_fitted_'), "Please fit the model first"

        if isinstance(X_train, pd.DataFrame):
            X_train = X_train.to_numpy()

        beta0 = self.beta0_
        betas = self.betas_

        clf = RegScoreRegressor(intercept=beta0, coefficients=betas, X_train=X_train)
        clf.reset_featureNames(feature_names)
        clf.print_model_card(unit=unit)
