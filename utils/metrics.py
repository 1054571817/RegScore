import math
import os

import numpy as np
import pandas as pd
from sklearn.feature_selection import r_regression
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error, \
    mean_absolute_percentage_error, r2_score, \
    accuracy_score, precision_score, recall_score, f1_score


def compute_ci(y_true, y_pred, func, n_bootstrap=100, ci=95, **kwargs):
    scores = []

    for i in range(n_bootstrap):
        indices = np.random.choice(range(len(y_true)), size=len(y_true), replace=True)
        y_true_resampled = y_true[indices]
        y_pred_resampled = y_pred[indices]

        scores.append(func(y_true_resampled, y_pred_resampled, **kwargs))

    lower_bound = np.percentile(scores, (100 - ci) / 2)
    upper_bound = np.percentile(scores, 100 - (100 - ci) / 2)

    return lower_bound, upper_bound


def compute_ci_detailed(metrics, stds, y_true, y_pred):
    n_samples = len(y_true)

    def l_u(mean, std, n_samples):
        ci = 1.96 * std / math.sqrt(n_samples)
        return mean - ci, mean + ci

    def custom_r_regression(y_true, y_pred):
        return r_regression(y_true.to_numpy().reshape(-1, 1), y_pred)[0]

    mae_ci = l_u(metrics[0], stds[0], n_samples)
    mape_ci = l_u(metrics[1], stds[1], n_samples)
    rmse_ci = compute_ci(y_true, y_pred, root_mean_squared_error)
    mse_ci = l_u(metrics[3], stds[3], n_samples)
    r2_ci = compute_ci(y_true, y_pred, r2_score)
    r_ci = compute_ci(y_true, y_pred, custom_r_regression)

    y_true[y_true <= 25] = 0
    y_true[y_true > 25] = 1
    y_pred[y_pred <= 25] = 0
    y_pred[y_pred > 25] = 1
    acc_ci = compute_ci(y_true, y_pred, accuracy_score)
    precision_ci = compute_ci(y_true, y_pred, precision_score)
    recall_ci = compute_ci(y_true, y_pred, recall_score)
    f1_ci = compute_ci(y_true, y_pred, f1_score)
    return ((
                mae_ci[0], mape_ci[0], rmse_ci[0], mse_ci[0], r2_ci[0], r_ci[0], acc_ci[0], precision_ci[0],
                recall_ci[0], f1_ci[0]),
            (mae_ci[1], mape_ci[1], rmse_ci[1], mse_ci[1], r2_ci[1], r_ci[1], acc_ci[1], precision_ci[1], recall_ci[1],
             f1_ci[1]))


def compute_metrics_detailed(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    r = r_regression(y_true.to_numpy().reshape(-1, 1), y_pred)[0]

    y_true[y_true <= 25] = 0
    y_true[y_true > 25] = 1
    y_pred[y_pred <= 25] = 0
    y_pred[y_pred > 25] = 1
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return mae, mape, rmse, mse, r2, r, acc, prec, rec, f1


def compute_stds(y_true, y_pred):
    mae_values = np.abs(y_true - y_pred)
    mae_std = np.std(mae_values)

    ape = np.abs((y_true - y_pred) / y_true)
    mape_std = np.std(ape)

    mse_values = (y_true - y_pred) ** 2
    mse_std = np.std(mse_values)
    return mae_std, mape_std, None, mse_std, None, None, None, None, None, None


def compute_by_fold_metrics(df_preds, gt_col="y", pred_col="y_pred"):
    all_metrics = []
    for f in df_preds["fold"].unique():
        y_true = df_preds[df_preds["fold"] == f][gt_col]
        y_pred = df_preds[df_preds["fold"] == f][pred_col]
        curr_metrics = compute_metrics_detailed(y_true, y_pred)
        all_metrics.append([f, *curr_metrics])
    header = ["FOLD", "MAE", "MAPE", "RMSE", "MSE", "R2", "R", "Acc", "Precision", "Recall", "F1"]
    metrics = pd.DataFrame(all_metrics, columns=header)
    if df_preds["fold"].nunique() != 1:
        mean = pd.DataFrame(metrics.mean()).T
        mean["FOLD"] = "MEAN"
        std = pd.DataFrame(metrics.std()).T
        std["FOLD"] = "STD"
        result = pd.concat([metrics, mean, std])
    else:
        stds = compute_stds(df_preds[gt_col], df_preds[pred_col])
        cis_l, cis_u = compute_ci_detailed(curr_metrics, stds, df_preds[gt_col], df_preds[pred_col])
        stds = pd.DataFrame([["STD", *stds]], columns=header)
        cis_l = pd.DataFrame([["CI_L", *cis_l]], columns=header)
        cis_u = pd.DataFrame([["CI_U", *cis_u]], columns=header)
        result = pd.concat([metrics, stds, cis_l, cis_u])
    return result


def compute_by_bins_metrics(df_preds, gt_col="y", pred_col="y_pred", bins=[20, 25, 30]):
    all_metrics = []
    bins_inf = [-99999, *bins, 99999]
    all_stds = []
    for f in df_preds["fold"].unique():
        for i in range(len(bins_inf) - 1):
            y_true = df_preds[
                (df_preds["fold"] == f) & (df_preds[gt_col] > bins_inf[i]) & (df_preds[gt_col] <= bins_inf[i + 1])][
                gt_col]
            y_pred = df_preds[
                (df_preds["fold"] == f) & (df_preds[gt_col] > bins_inf[i]) & (df_preds[gt_col] <= bins_inf[i + 1])][
                pred_col]
            if len(y_pred) == 0:
                continue
            metrics = compute_metrics_detailed(y_true, y_pred)
            if i == 0:
                bin_text = f"y<={bins_inf[i + 1]}"
            elif i == (len(bins_inf) - 2):
                bin_text = f"{bins_inf[i]}<y"
            else:
                bin_text = f"{bins_inf[i]}<y<={bins_inf[i + 1]}"
            if df_preds["fold"].nunique() == 1:
                stds = compute_stds(y_true, y_pred)
                all_stds.append(["STD", bin_text, *stds])
            all_metrics.append([f, bin_text, *metrics])
    header = ["FOLD", "BIN", "MAE", "MAPE", "RMSE", "MSE", "R2", "R", "Acc", "Precision", "Recall", "F1"]
    metrics = pd.DataFrame(all_metrics, columns=header)
    if df_preds["fold"].nunique() != 1:
        mean = metrics.groupby("BIN").mean()
        mean["FOLD"] = "MEAN"
        std = metrics.groupby("BIN").std()
        std["FOLD"] = "STD"
        result = pd.concat([metrics, mean, std])
    else:
        std = pd.DataFrame(all_stds, columns=header)
        result = pd.concat([metrics, std])
    return result


def compute_total_metrics(by_fold_metrics, n_samples, model_name=""):
    columns_to_multiply = ["MAPE", "R", "Acc", "Precision", "Recall", "F1"]
    by_fold_metrics[columns_to_multiply] = by_fold_metrics[columns_to_multiply] * 100
    if by_fold_metrics["FOLD"].nunique() == 4:  # only one fold (fold + std)
        mean = by_fold_metrics[by_fold_metrics["FOLD"] == 0]
        std = by_fold_metrics[by_fold_metrics["FOLD"] == "STD"]
        ci_l = by_fold_metrics[by_fold_metrics["FOLD"] == "CI_L"]
        ci_u = by_fold_metrics[by_fold_metrics["FOLD"] == "CI_U"]
    else:
        mean = by_fold_metrics[by_fold_metrics["FOLD"] == "MEAN"]
        std = by_fold_metrics[by_fold_metrics["FOLD"] == "STD"]
        ci_l = by_fold_metrics[by_fold_metrics["FOLD"] == "CI_L"]
        ci_u = by_fold_metrics[by_fold_metrics["FOLD"] == "CI_U"]
    # mean.loc[:, "MAPE"] = mean["MAPE"] * 100
    # std.loc[:, "MAPE"] = std["MAPE"] * 100

    std = std.drop(columns="FOLD")
    mean = mean.drop(columns="FOLD")

    if not ci_l.empty:
        # ci_l.loc[:, "MAPE"] = ci_l["MAPE"] * 100
        # ci_u.loc[:, "MAPE"] = ci_u["MAPE"] * 100
        ci_l = ci_l.drop(columns="FOLD")
        ci_u = ci_u.drop(columns="FOLD")

        total_metrics = mean.round(2).astype(str) + "+/-" + std.round(2).astype(str) + " (" + ci_l.round(2).astype(
            str) + ", " + ci_u.round(2).astype(str) + ")"
    else:
        total_metrics = mean.round(2).astype(str) + " \\pm " + std.round(2).astype(str)
    total_metrics = total_metrics.apply(lambda x: x.str.replace("+/-nan", "", regex=False))
    total_metrics["MODEL"] = model_name
    return total_metrics


def compute_metrics(df_preds, gt_col="y", pred_col="y_pred", bins=[20, 25, 30], model_name=""):
    n_samples = len(df_preds)
    by_fold_metrics = compute_by_fold_metrics(df_preds, gt_col, pred_col)
    by_bins_metrics = compute_by_bins_metrics(df_preds, gt_col, pred_col, bins)
    total_metrics = compute_total_metrics(by_fold_metrics, n_samples, model_name)
    return total_metrics, by_fold_metrics, by_bins_metrics


def save_metrics(total_metrics, by_fold_metrics, by_bins_metrics, model_name, folder_results):
    total_metrics.to_csv(os.path.join(folder_results, f"{model_name}_total_model_metrics.csv"))
    by_fold_metrics.to_csv(os.path.join(folder_results, f"{model_name}_by_fold_metrics.csv"))
    by_bins_metrics.to_csv(os.path.join(folder_results, f"{model_name}_by_bins_metrics.csv"))


if __name__ == '__main__':
    df = pd.DataFrame({
        'fold': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
        'gt': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'pred': [1.1, 1.9, 3.2, 4.1, 4.8, 1.1, 1.9, 3.2, 4.1, 4.8, 1.1, 1.9, 3.2, 4.1, 4.8]
    })
    t_m, bf_m, bb_m = compute_metrics(df, "gt", "pred")
    print(t_m)
