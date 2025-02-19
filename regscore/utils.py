import numpy as np
from itertools import product
import requests


def get_groupIndex_to_featureIndices(featureIndex_to_groupIndex):
    groupIndex_to_featureIndices = {}
    for featureIndex, groupIndex in enumerate(featureIndex_to_groupIndex):
        if groupIndex not in groupIndex_to_featureIndices:
            groupIndex_to_featureIndices[groupIndex] = set()
        groupIndex_to_featureIndices[groupIndex].add(featureIndex)
    return groupIndex_to_featureIndices


def get_support_indices(betas):
    return np.where(np.abs(betas) > 1e-9)[0]


def get_nonsupport_indices(betas):
    return np.where(np.abs(betas) <= 1e-9)[0]


def normalize_X(X):
    X_mean = np.mean(X, axis=0)
    X_norm = np.linalg.norm(X - X_mean, axis=0)
    scaled_feature_indices = np.where(X_norm >= 1e-9)[0]
    X_normalized = X - X_mean
    X_normalized[:, scaled_feature_indices] = X_normalized[:, scaled_feature_indices] / X_norm[[scaled_feature_indices]]
    return X_normalized, X_mean, X_norm, scaled_feature_indices


def compute_logisticLoss_from_yXB(yXB):
    # shape of yXB is (n, )
    return np.sum(np.log(1. + np.exp(-yXB)))


def compute_logisticLoss_from_ExpyXB(ExpyXB):
    # shape of ExpyXB is (n, )
    return np.sum(np.log(1. + np.reciprocal(ExpyXB)))


def convert_y_to_neg_and_pos_1(y):
    y_max, y_min = np.min(y), np.max(y)
    y_transformed = -1 + 2 * (y - y_min) / (y_max - y_min)  # convert y to -1 and 1
    return y_transformed


def isEqual_upTo_8decimal(a, b):
    if np.isscalar(a):
        return abs(a - b) < 1e-8
    return np.max(np.abs(a - b)) < 1e-8


def isEqual_upTo_16decimal(a, b):
    if np.isscalar(a):
        return abs(a - b) < 1e-16
    return np.max(np.abs(a - b)) < 1e-16


def insertIntercept_asFirstColOf_X(X):
    n = len(X)
    intercept = np.ones((n, 1))
    X_with_intercept = np.hstack((intercept, X))
    return X_with_intercept


def get_all_product_booleans(sparsity=5):
    # build list of lists:
    all_lists = []
    for i in range(sparsity):
        all_lists.append([0, 1])
    all_products = list(product(*all_lists))
    all_products = [list(elem) for elem in all_products]
    return np.array(all_products)


def get_continuousFeatureName_from_binaryFeatureName(binaryFeatureName):
    # convert binary feature name to continuous feature name
    # check whether binaryFeatureName is in the format 'FeatureName<=Threshold' or 'Threshold1<FeatureName<=Threshold2'
    featureName = None
    errorMessage = f"Feature name {binaryFeatureName} does not follow the format 'FeatureName<=Threshold' or 'Threshold1<FeatureName<=Threshold2' or FeatureName_isNaN!"

    if '_isNaN' in binaryFeatureName:
        num_isnan = binaryFeatureName.count('_isNaN')
        if num_isnan > 1:
            raise ValueError(f"{errorMessage}")
        if binaryFeatureName[-6:] != '_isNaN':
            raise ValueError(f"{errorMessage}")
        featureName = binaryFeatureName.split('_')[0]

    elif '<=' in binaryFeatureName:
        num_leq = binaryFeatureName.count('<=')
        num_less = binaryFeatureName.count('<')

        if num_less == 2 and num_leq == 1:
            # this is the case where the feature name is in the form of 'Threshold1<FeatureName<=Threshold2'
            try:
                featureName = binaryFeatureName.split('<')[1].split('<=')[0]
            except:
                raise ValueError(f"{errorMessage}")

        elif num_less == 1 and num_leq == 1:
            # this is the case where the feature name is in the form of 'FeatureName<=Threshold'
            try:
                featureName = binaryFeatureName.split('<=')[0]
            except:
                raise ValueError(f"{errorMessage}")

        else:
            raise ValueError(f"{errorMessage}")

    else:
        raise ValueError(f"{errorMessage}")

    if not isinstance(featureName, str):
        raise ValueError(f"{errorMessage}")

    featureName = featureName.strip()
    if len(featureName) == 0:
        raise ValueError(f"{errorMessage}")

    return featureName


def get_groupIndex_from_featureNames(featureNames):
    # from a list of feature names, get the group index for each feature
    print(
        "We convert binary feature names to continuous feature names\nNote that the continuous feature names should be in the form of 'FeatureName<=Threshold' or 'Threshold1<FeatureName<=Threshold2' or 'FeatureName_isNaN'!\nFor datasets from RiskSLIM (https://github.com/ustunb/risk-slim/tree/master/examples/data), we hardcode the conversion since the feature names do not follow the above format.")

    groupIndex = check_if_featureNames_come_from_RiskSLIM_GitHub_data(featureNames)
    if len(groupIndex) > 0:
        return np.asarray(groupIndex, dtype=int)

    continuousFeatureNameIndexDict = dict()
    numContinuousFeatures = 0

    for featureName in featureNames:
        continuousFeatureName = get_continuousFeatureName_from_binaryFeatureName(featureName)
        if continuousFeatureName not in continuousFeatureNameIndexDict:
            continuousFeatureNameIndexDict[continuousFeatureName] = numContinuousFeatures
            numContinuousFeatures += 1
        groupIndex.append(continuousFeatureNameIndexDict[continuousFeatureName])

    return np.asarray(groupIndex, dtype=int)


def compute_mseLoss(residuals, lambda2, betas):
    """
    Compute the mean squared error (MSE) loss with L2 regularization.

    Parameters:
    - residuals: Residuals of the model (y - X\beta).
    - lambda2: Regularization parameter (L2 penalty).
    - betas: Model coefficients.

    Returns:
    - Loss: MSE loss with L2 regularization.
    """
    mse_loss = np.mean(residuals ** 2)  # Mean squared error
    l2_penalty = lambda2 * np.sum(betas ** 2)  # L2 regularization term
    return mse_loss + l2_penalty
