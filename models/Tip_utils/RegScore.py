import json
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.Tip_utils.pieces import DotDict


class AdaptiveTopKGating(nn.Module):
    def __init__(self, k, num_features, bin_features, tau: float = 0.1):
        """
        Initialize the Top-K Gating module with per-sample scores.
        :param k: Number of top features to select.
        :param num_features: Number of original features.
        :param bin_features: Number of binned features.
        :param tau: Temperature for sigmoid approximation.
        """
        super(AdaptiveTopKGating, self).__init__()
        self.k = k
        self.tau = tau

        self.binning_map = nn.Linear(num_features, bin_features, bias=False)

    def forward(self, x, x_emb):
        """
        Forward pass for the gating module.
        :param x: Input tensor of shape (batch_size, bin_features).
        :param x_emb: Embedding tensor of shape (batch_size, features, embedding_dim).
        :return: A differentiable mask of shape (batch_size, bin_features).
        """

        mask_hard, mask_soft = self.get_mask(x_emb)

        mask = mask_hard.detach() - mask_soft.detach() + mask_soft

        return mask * x

    def get_mask(self, x_emb):
        """
        Get the hard mask for a given embedding.
        :param x_emb: Embedding tensor of shape (batch_size, features, embedding_dim).
        :return: A hard mask of shape (batch_size, bin_features).
        """
        learnable_scores = x_emb.mean(dim=-1)  # Compute learnable scores (batch_size, features)

        # Map learnable_scores into bin_features space using the linear layer
        aggregated_scores = self.binning_map(learnable_scores)  # (batch_size, bin_features)

        thresholds = torch.topk(aggregated_scores, self.k, dim=-1, largest=True, sorted=False).values[:, -1]
        thresholds = thresholds.unsqueeze(-1)  # (batch_size, 1)

        mask_hard = (aggregated_scores >= thresholds).float()  # Hard mask (batch_size, bin_features)
        mask_soft = torch.sigmoid((aggregated_scores - thresholds) / self.tau)  # Soft mask for backward pass
        return mask_hard, mask_soft


class PersonalizedRegScore(nn.Module):
    """

    """

    def __init__(self, args: Dict, cat_lengths_tabular: List, con_lengths_tabular: List) -> None:
        super(PersonalizedRegScore, self).__init__()

        self.feature_names_p = args.feature_names
        self.feature_norms_p = args.feature_norms
        self.bin_feature_names_p = args.binarized_column_names
        self.bin_features_static = args.bin_features_static
        self.bin_column_lengths_p = args.binarized_column_lengths
        self.binning_strategy = args.binning_strategy

        self.num_cat = len(cat_lengths_tabular)
        self.num_con = len(con_lengths_tabular)
        self.num_unique_cat = sum(cat_lengths_tabular)
        self.cat_lengths_tabular = cat_lengths_tabular
        self.con_lengths_tabular = con_lengths_tabular
        self.hidden_dim = args.multimodal_embedding_dim

        self.bin_features = self.bin_features_static
        self.model_size_m = args.model_size_m
        self.masking = AdaptiveTopKGating(self.model_size_m, self.num_cat + self.num_con, self.bin_features)
        self.emb_to_reg = nn.Linear(self.hidden_dim, 1 + self.bin_features)
        self.register_buffer('bias', torch.Tensor([args.median]))
        if args.checkpoint is None:
            self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            m.weight.data.normal_(mean=0.0, std=.02)
        elif isinstance(m, nn.LayerNorm):
            m.bias.data.zero_()
            m.weight.data.fill_(1.0)
        if isinstance(m, nn.Linear) and m.bias is not None:
            m.bias.data.zero_()

    def _resolve_feature_names(self, bin_edges):
        """
        Generate new feature names for one-hot encoded categorical variables
        and binned continuous variables.

        Parameters:
            feature_names (list): Original feature names.
            cat_lengths_tabular (list): List of the number of unique categories for each categorical feature.
            con_lengths_tabular (list): List of the number of bins for each continuous feature.
            bin_edges (list of numpy arrays): List of bin edges for continuous features.

        Returns:
            list: New feature names with one-hot and bin-based descriptions.
        """
        new_feature_names = []
        cat_idx = 0
        con_idx = 0

        # Process categorical features
        for length in self.cat_lengths_tabular:
            cat_feature_name = self.feature_names[cat_idx]
            for i in range(length):
                new_feature_names.append(f"{cat_feature_name}={i}")
            cat_idx += 1

        for bin_edges_feature in bin_edges:
            con_feature_name = self.feature_names[len(self.cat_lengths_tabular) + con_idx]
            for i in range(len(bin_edges_feature) - 1):
                lower = bin_edges_feature[i]
                upper = bin_edges_feature[i + 1]
                new_feature_names.append(f"{lower}<={con_feature_name}<{upper}")
            con_idx += 1

        return new_feature_names

    def test_epoch_end(self):
        with open(self.feature_names_p, "r") as f:
            self.feature_names = json.load(f)

        with open(self.bin_feature_names_p, "r") as f:
            self.bin_feature_names = json.load(f)
        con_stats = np.load(self.feature_norms_p)
        self.con_means = con_stats["mean"]
        self.con_std = con_stats["std"]

        batch_size = self.reg_params.shape[0]
        dfs = []
        for i in range(batch_size):
            reg_params_sample = self.reg_params[i].squeeze().numpy().tolist()
            if self.binning_strategy != "static":
                bin_edges_sample = self.bin_edges[i].numpy()

                bin_edges_readable = bin_edges_sample * np.expand_dims(self.con_std, -1) + np.expand_dims(
                    self.con_means,
                    -1)
                bin_edges_readable = np.round(bin_edges_readable, 2)
                scoring_feature_names = self._resolve_feature_names(bin_edges_readable)
            else:
                scoring_feature_names = self.bin_feature_names

            mask = self.masking.get_mask(self.x_emb[:, 1:, :])[0][i].detach().cpu().numpy().tolist()

            features = ["bias"] + [scoring_feature_names[int(j)] for j, m in enumerate(mask) if m == 1]
            reg_params_sample = [reg_params_sample[0]] + \
                                [reg_params_sample[int(j + 1)] for j, m in enumerate(mask) if m == 1]

            df = pd.DataFrame({
                "Feature": features,
                "Score": reg_params_sample
            })
            dfs.append(df)

        for i, df in enumerate(dfs):
            print(f"Sample {i + 1}:\n", df)
        return df

    def forward(self, x_t: torch.Tensor, x_m_emb: torch.Tensor, x_bin: torch.Tensor) -> torch.Tensor:
        bias_column = torch.ones((x_t.shape[0], 1), device=x_t.device)
        if self.binning_strategy != "static":
            x_cat = self.one_hot(x_t[:, :self.num_cat].long())
            x_con_bin, bin_edges = self.binning(x_t[:, self.num_cat:], x_m_emb[:, 1 + self.num_cat:])
            x_t_bin = torch.cat([x_cat, x_con_bin], dim=1)
        else:
            x_t_bin = x_bin
        x_t_bin = self.masking(x_t_bin, x_m_emb[:, 1:, :])
        x_t_b = torch.cat([bias_column, x_t_bin], dim=1)
        reg_params = self.emb_to_reg(x_m_emb[:, 0, :])

        out = torch.sum(x_t_b * reg_params, dim=1, keepdim=True)

        if not self.training:
            self.reg_params = reg_params.detach().cpu()
            self.x_emb = x_m_emb
            if self.binning_strategy != "static":
                self.bin_edges = bin_edges.detach().cpu()

        return out


if __name__ == '__main__':
    m = PersonalizedRegScore(
        DotDict(
            {'checkpoint': None, 'multimodal_embedding_dim': 512, 'binning_k': 3, 'model_size_m': 10, 'median': 40,
             'binning_strategy': "static", "bin_features_static": 86}),
        [3],
        [1 for i in range(34)])
    x_t = torch.tensor([[2.0, 3.0, 0.0, 2.0, 0.2, -0.1, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, 0.1,
                         3.0, 0.0, 2.0, 0.2, -0.1, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, 0.1, 0.2,
                         -0.2],
                        [1.0, 1.0, 1.0, 0.0, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, 0.1, 3.0,
                         0.0, 2.0, 0.2, -0.1, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, -0.5, 0.2, 0.1, 0.2, 0.1]],
                       dtype=torch.float32)
    x_m_emb = torch.randn((2, 36, 512))
    x_bin = torch.ones(2, 86)
    print(m(x_t, x_m_emb, x_bin))
    print(m(x_t, x_m_emb, x_bin).shape)
