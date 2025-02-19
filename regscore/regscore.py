import numpy as np
import sklearn.metrics

from regscore.sparseBeamSearch import sparseLinRegModel
from regscore.utils import get_support_indices, get_all_product_booleans


class RegScoreOptimizer:
    def __init__(self, X, y, k, select_top_m=50, gap_tolerance=0.05, parent_size=10, child_size=None,
                 featureIndex_to_groupIndex=None):
        """Initialize the RegScoreOptimizer class, which performs sparseBeamSearch and generates integer sparseDiverseSet

        Parameters
        ----------
        X : ndarray
            (2D array with `float` type) feature matrix, each row[i, :] corresponds to the features of sample i
        y : ndarray
            (1D array with `float` type) labels (+1 or -1) of each sample
        k : int
            number of selected features in the final sparse model
        select_top_m : int, optional
            number of top solutions to keep among the pool of diverse sparse solutions, by default 50
        parent_size : int, optional
            how many solutions to retain after beam search, by default 10
        child_size : int, optional
            how many new solutions to expand for each existing solution, by default None
        featureIndex_to_groupIndex : ndarray, optional
            (1D array with `int` type) featureIndex_to_groupIndex[i] is the group index of feature i, by default None
        """

        y_shape = y.shape
        X_shape = X.shape
        assert len(y_shape) == 1, "input y must have 1-D shape!"
        assert len(X_shape) == 2, "input X must have 2-D shape!"
        assert X_shape[0] == y_shape[
            0], "number of rows from input X must be equal to the number of elements from input y!"
        self.y = y
        self.X = X

        self.k = k
        self.parent_size = parent_size
        self.child_size = self.parent_size
        if child_size is not None:
            self.child_size = child_size

        self.sparseDiverseSet_gap_tolerance = gap_tolerance
        self.sparseDiverseSet_select_top_m = select_top_m

        self.featureIndex_to_groupIndex = featureIndex_to_groupIndex

        self.multipliers = None
        self.sparseDiversePool_beta0_integer = None
        self.sparseDiversePool_betas_integer = None

        self.sparseLogRegModel_object = sparseLinRegModel(X, y, intercept=True)

        self.IntegerPoolIsSorted = False

    def optimize(self):
        """performs sparseBeamSearch
        """
        self.sparseLogRegModel_object.get_sparse_sol_via_OMP(k=self.k, parent_size=self.parent_size,
                                                             child_size=self.child_size)

        self.beta0, self.betas = self.sparseLogRegModel_object.get_beta0_betas()


class RegScoreRegressor:
    def __init__(self, intercept, coefficients, featureNames=None, X_train=None):
        """Initialize a RegScore regressor. Then we can use this classifier to predict

        Parameters
        ----------
        multiplier : float
            multiplier of the risk score model
        intercept : float
            intercept of the risk score model
        coefficients : ndarray
            (1D array with `float` type) coefficients of the regscore model
        """
        self.intercept = intercept
        self.coefficients = coefficients

        self.X_train = X_train

        self.reset_featureNames(featureNames)

    def predict(self, X):
        """Predict values

        Parameters
        ----------
        X : ndarray
            (2D array with `float` type) feature matrix with shape (n, p)

        Returns
        -------
        y_pred : ndarray
            (1D array with `float` type) predicted values with shape (n, )
        """
        y_score = (self.intercept + X.dot(self.coefficients))
        return y_score

    def reset_featureNames(self, featureNames):
        """Reset the feature names in the class in order to print out the model card for the user

        Parameters
        ----------
        featureNames : str[:]
            a list of strings which are the feature names for columns of X
        """
        self.featureNames = featureNames

    def _print_score_calculation_table(self, unit="point(s)"):
        assert self.featureNames is not None, "please pass the featureNames to the model by using the function .reset_featureNames(featureNames)"

        nonzero_indices = get_support_indices(self.coefficients)

        max_feature_length = max([len(featureName) for featureName in self.featureNames])
        row_score_template = '{0}. {1:>%d}     {2:.2f} %s | + ...' % (max_feature_length, unit)

        print("The RegScore is:")
        row_score_str = row_score_template.format("0", "bias", self.intercept)
        print(row_score_str)
        for count, feature_i in enumerate(nonzero_indices):
            row_score_str = row_score_template.format(count + 1, self.featureNames[feature_i],
                                                      self.coefficients[feature_i])
            if count == 0:
                row_score_str = row_score_str.replace("+", " ")

            print(row_score_str)

        final_score_str = ' ' * (14 + max_feature_length) + 'SCORE | =    '
        print(final_score_str)

    def _print_score_risk_row(self, scores, risks):
        score_row = "SCORE |"
        risk_row = "RISK  |"
        score_entry_template = '  {0:>4}  |'
        risk_entry_template = ' {0:>5}% |'
        for (score, risk) in zip(scores, risks):
            score_row += score_entry_template.format(score)
            risk_row += risk_entry_template.format(round(100 * risk, 1))
        print(score_row)
        print(risk_row)

    def print_model_card(self, unit="point(s)"):
        """Print the score evaluation table and score risk table onto terminal
        """
        self._print_score_calculation_table(unit=unit)
