import os
import pandas as pd
from metrics import compute_metrics, save_metrics
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor


def split_folds(df, y_col="y", fold_col="test_set_fold"):
    X_trains = []
    y_trains = []
    y = df[y_col]
    test_set_fold = df[fold_col]
    X = df.drop(columns=[y_col, fold_col])
    X_test = X[test_set_fold == -1]
    y_test = y[test_set_fold == -1]
    X = X[test_set_fold != -1]
    y = y[test_set_fold != -1]
    test_set_fold = test_set_fold[test_set_fold != -1]
    n_folds = test_set_fold.nunique() - 1 if test_set_fold.nunique() == 2 else test_set_fold.nunique()
    for i in range(n_folds):
        X_trains.append(X[test_set_fold != i])
        y_trains.append(y[test_set_fold != i])
    return X_trains, y_trains, X_test, y_test


def train_predict(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions


def build_df_fold_predictions(fold, y, predictions):
    return pd.DataFrame(index=y.index, data={"fold": [fold] * len(y), "y": y, "y_pred": predictions})


def get_df(path):
    df = pd.read_csv(path, index_col=0)
    df.loc[df["pa_mean"] <= 25, "final_primary_ph_diagnosis"] = 0
    df.loc[df["final_primary_ph_diagnosis"] != 0, "final_primary_ph_diagnosis"] = 1
    cols_to_drop = ["phd"]
    df = df.drop(columns=cols_to_drop)

    categorical_cols = ["pericardial_effusion"]
    continuous_cols = ['heart_rate', 'who_functional_class', 'reveal_score',
                       'reveal_score_lite', 'compera_score', 'lvedv', 'lvedv_index', 'lvesv', 'lvesv_index', 'lvsv',
                       'lvsv_index', 'lvco',
                       'lvef', 'vmi', 'rvedv', 'rvesv', 'rvsv', 'rvedv_index', 'rvesv_index',
                       'rvsv_index', 'rvef', 'rv_dia_mass_index', 'rv_syst_mass',
                       'manual_la_volume', 'manual_la_volume_index', 'manual_4ch_la_area',
                       'manual_2ch_la_length', 'pa_relative_area_change', 'diastolic_pa_area',
                       'systolic_pa_area', 'pa_forward_flow_per_min',
                       'pa_backward_flow_per_min', 'aa_forward_flow_per_min',
                       'septal_angle_syst', 'septal_angle_diast']
    categorical_df = df.loc[:, categorical_cols]
    continuous_df = df.loc[:, continuous_cols]
    continuous_df = (continuous_df - continuous_df.mean()) / continuous_df.std()

    df = pd.concat([df[["pa_mean", "test_set_fold"]], categorical_df, continuous_df], axis=1)
    return df


def compute_ml_methods(path_data, folder_results):
    df = get_df(path_data)
    X_trains, y_trains, X_test, y_test = split_folds(df, y_col="pa_mean")
    models = [LinearRegression, GradientBoostingRegressor, RandomForestRegressor, MLPRegressor, XGBRegressor, DecisionTreeRegressor]
    args = [{}, {"random_state": 42}, {"random_state": 42}, {"random_state": 42, 'max_iter': 1000},
            {"random_state": 42}, {}]
    total_results = None
    for model_cls, arg in zip(models, args):
        fold_predictions = pd.DataFrame(columns=["fold", "y", "y_pred"])
        for i, (X_train, y_train) in enumerate(zip(X_trains, y_trains)):
            model = model_cls(**arg)
            predictions = train_predict(model, X_train, y_train, X_test)
            current_fold_predictions = build_df_fold_predictions(i, y_test, predictions)
            if fold_predictions.empty:
                fold_predictions = current_fold_predictions
            else:
                fold_predictions = pd.concat([fold_predictions, current_fold_predictions])
        total_model_metrics, by_fold_metrics, by_bins_metrics = compute_metrics(fold_predictions,
                                                                                model_name=model_cls.__name__)
        model_name = model_cls.__name__
        print(model_name)
        print(total_model_metrics)
        print(by_fold_metrics)
        print(by_bins_metrics)
        print("_______________")
        if total_results is None:
            total_results = total_model_metrics
        else:
            total_results = pd.concat([total_results, total_model_metrics])
        save_metrics(total_model_metrics, by_fold_metrics, by_bins_metrics, model_name, folder_results)
    print(total_results)
    total_results.to_csv(os.path.join(folder_results, "ph_reg_ml_results_5_fold.csv"), index=False)


if __name__ == '__main__':
    path_data = "/data/ph/ph_orig_5_fold.csv"
    folder_results = "/ml_output"
    compute_ml_methods(path_data, folder_results)
