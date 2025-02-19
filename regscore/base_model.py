import numpy as np
from regscore.utils import normalize_X, compute_mseLoss


class linRegModel:
    def __init__(self, X, y, lambda2=1e-8, intercept=True):
        self.X = X
        self.X_normalized, self.X_mean, self.X_norm, self.scaled_feature_indices = normalize_X(self.X)
        self.n, self.p = self.X_normalized.shape
        self.y = y.reshape(-1).astype(float)
        self.yX = self.y.reshape(-1, 1) * self.X_normalized
        self.yXT = np.zeros((self.p, self.n))
        self.yXT[:] = np.transpose(self.yX)[:]
        self.beta0 = 0
        self.betas = np.zeros((self.p,))

        self.intercept = intercept
        self.lambda2 = lambda2
        self.twoLambda2 = 2 * self.lambda2

        # Approximated bounded Lipschitz constant for MSE loss
        self.Lipschitz = (1 / self.n) * np.linalg.norm(self.X_normalized.T @ self.X_normalized, ord=2) + self.twoLambda2

        self.total_child_added = 0

    def warm_start_from_original_beta0_betas(self, original_beta0, original_betas):
        self.original_beta0 = original_beta0
        self.original_betas = original_betas
        self.beta0, self.betas = self.transform_coefficients_to_normalized_space(self.original_beta0,
                                                                                 self.original_betas)

    def warm_start_from_beta0_betas(self, beta0, betas):
        self.beta0, self.betas = beta0, betas

    def get_beta0_betas(self):
        return self.beta0, self.betas

    def get_original_beta0_betas(self):
        return self.transform_coefficients_to_original_space(self.beta0, self.betas)

    def transform_coefficients_to_original_space(self, beta0, betas):
        original_betas = betas.copy()
        original_betas[self.scaled_feature_indices] /= self.X_norm[self.scaled_feature_indices]
        original_beta0 = beta0 - np.dot(self.X_mean, original_betas)
        return original_beta0, original_betas

    def transform_coefficients_to_normalized_space(self, original_beta0, original_betas):
        betas = original_betas.copy()
        betas[self.scaled_feature_indices] *= self.X_norm[self.scaled_feature_indices]
        beta0 = original_beta0 + self.X_mean.dot(original_betas)
        return beta0, betas

    def get_grad_at_coord(self, residuals, betas_j, X_j):
        return -2 * np.dot(X_j, residuals) / self.n + self.twoLambda2 * betas_j

    def update_residuals(self, residuals, X_j, diff_betas_j):
        residuals -= X_j * diff_betas_j

    def optimize_1step_at_coord(self, residuals, betas, X_j, j):
        prev_betas_j = betas[j]
        grad_at_j = self.get_grad_at_coord(residuals, prev_betas_j, X_j)
        step_at_j = grad_at_j / self.Lipschitz
        current_betas_j = prev_betas_j - step_at_j
        betas[j] = current_betas_j
        diff_betas_j = betas[j] - prev_betas_j
        self.update_residuals(residuals, X_j, diff_betas_j)

    def finetune_on_current_support(self, residuals, beta0, betas, total_CD_steps=100):
        support = np.where(np.abs(betas) > 1e-9)[0]
        loss_before = compute_mseLoss(residuals, self.lambda2, betas[support])

        for steps in range(total_CD_steps):
            if self.intercept:
                grad_intercept = -2 * np.sum(residuals) / self.n
                step_at_intercept = grad_intercept
                beta0 -= step_at_intercept
                residuals += step_at_intercept

            for j in support:
                self.optimize_1step_at_coord(residuals, betas, self.X_normalized[:, j], j)

            if steps % 10 == 0:
                loss_after = compute_mseLoss(residuals, self.lambda2, betas[support])
                if abs(loss_before - loss_after) / loss_after < 1e-8:
                    break
                loss_before = loss_after

        return residuals, beta0, betas

    def compute_residuals(self, beta0, betas):
        return self.y - (beta0 + np.dot(self.X_normalized, betas))
