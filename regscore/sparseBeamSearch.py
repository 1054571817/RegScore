import numpy as np
from regscore.utils import get_support_indices, get_nonsupport_indices
from regscore.base_model import linRegModel, compute_mseLoss
from tqdm import tqdm

class sparseLinRegModel(linRegModel):
    def __init__(self, X, y, lambda2=1e-8, intercept=True):
        super().__init__(X=X, y=y, lambda2=lambda2, intercept=intercept)

    def getAvailableIndices_for_expansion(self, betas):
        """Get the indices of features that can be added to the support of the current sparse solution

        Parameters
        ----------
        betas : ndarray
            (1D array with `float` type) The current sparse solution

        Returns
        -------
        available_indices : ndarray
            (1D array with `int` type) The indices of features that can be added to the support of the current sparse solution
        """
        available_indices = get_nonsupport_indices(betas)
        return available_indices

    def expand_parent_i_support_via_OMP_by_1(self, i, child_size=10):
        """For parent solution i, generate [child_size] child solutions

        Parameters
        ----------
        i : int
            index of the parent solution
        child_size : int, optional
            how many child solutions to generate based on parent solution i, by default 10
        """
        non_support = self.getAvailableIndices_for_expansion(self.betas_arr_parent[i])
        support = get_support_indices(self.betas_arr_parent[i])

        residuals = self.compute_residuals(self.beta0_arr_parent[i], self.betas_arr_parent[i])
        grad_on_non_support = -2 * self.X_normalized[:, non_support].T.dot(residuals) / self.n
        abs_grad_on_non_support = np.abs(grad_on_non_support)

        num_new_js = min(child_size, len(non_support))
        new_js = non_support[np.argsort(-abs_grad_on_non_support)][:num_new_js]
        child_start, child_end = i * child_size, i * child_size + num_new_js

        self.residuals_arr_child[child_start:child_end] = residuals  # Shared initial residuals
        self.betas_arr_child[child_start:child_end] = 0
        self.betas_arr_child[child_start:child_end, support] = self.betas_arr_parent[i, support]
        self.beta0_arr_child[child_start:child_end] = self.beta0_arr_parent[i]

        beta_new_js = np.zeros((num_new_js, ))
        diff_max = 1e3

        step = 0
        while step < 10 and diff_max > 1e-3:
            prev_beta_new_js = beta_new_js.copy()
            grad_on_new_js = -2 * np.sum(self.X_normalized[:, new_js] * residuals.reshape(-1, 1), axis=0) / self.n + self.twoLambda2 * beta_new_js
            step_at_new_js = grad_on_new_js / self.Lipschitz

            beta_new_js = prev_beta_new_js - step_at_new_js
            diff_beta_new_js = beta_new_js - prev_beta_new_js

            for j, diff in zip(new_js, diff_beta_new_js):
                residuals -= self.X_normalized[:, j] * diff

            diff_max = np.max(np.abs(diff_beta_new_js))
            step += 1

        for l in range(num_new_js):
            child_id = child_start + l
            self.betas_arr_child[child_id, new_js[l]] = beta_new_js[l]
            tmp_support_str = str(get_support_indices(self.betas_arr_child[child_id]))
            if tmp_support_str not in self.forbidden_support:
                self.total_child_added += 1  # Count how many unique child has been added for a specified support size
                self.forbidden_support.add(tmp_support_str)

                self.residuals_arr_child[child_id], self.beta0_arr_child[child_id], self.betas_arr_child[child_id] = self.finetune_on_current_support(
                    self.residuals_arr_child[child_id], self.beta0_arr_child[child_id], self.betas_arr_child[child_id]
                )
                self.loss_arr_child[child_id] = compute_mseLoss(
                    self.residuals_arr_child[child_id], self.lambda2, self.betas_arr_child[child_id]
                )
                # print(self.loss_arr_child[child_id])

    def beamSearch_multipleSupports_via_OMP_by_1(self, parent_size=10, child_size=10):
        """Each parent solution generates [child_size] child solutions, so there will be [parent_size] * [child_size] number of total child solutions. However, only the top [parent_size] child solutions are retained as parent solutions for the next level i+1.

        Parameters
        ----------
        parent_size : int, optional
            how many top solutions to retain at each level, by default 10
        child_size : int, optional
            how many child solutions to generate based on each parent solution, by default 10
        """
        self.loss_arr_child.fill(1e12)
        self.total_child_added = 0

        for i in range(self.num_parent):
            self.expand_parent_i_support_via_OMP_by_1(i, child_size=child_size)

        child_indices = np.argsort(self.loss_arr_child)[:min(parent_size, self.total_child_added)]  # Get indices of children with smallest losses
        num_child_indices = len(child_indices)
        self.residuals_arr_parent[:num_child_indices], self.beta0_arr_parent[:num_child_indices], self.betas_arr_parent[:num_child_indices] = (
            self.residuals_arr_child[child_indices], self.beta0_arr_child[child_indices], self.betas_arr_child[child_indices]
        )

        self.num_parent = num_child_indices

    def get_sparse_sol_via_OMP(self, k, parent_size=10, child_size=10):
        """Get sparse solution through beam search and orthogonal matching pursuit (OMP). Each parent solution generates [child_size] child solutions, so there will be [parent_size] * [child_size] total child solutions. However, only the top [parent_size] child solutions are retained as parent solutions for the next level i+1.

        Parameters
        ----------
        k : int
            Number of nonzero coefficients for the final sparse solution
        parent_size : int, optional
            How many top solutions to retain at each level, by default 10
        child_size : int, optional
            How many child solutions to generate based on each parent solution, by default 10
        """
        num_nonzero = len(np.where(np.abs(self.betas) > 1e-9)[0])

        if num_nonzero == self.p:
            return

        if self.intercept and num_nonzero == 0:
            self.beta0 = np.mean(self.y)
            self.residuals = self.y - self.beta0

        # Create beam search parent
        self.residuals_arr_parent = np.zeros((parent_size, self.n))
        self.beta0_arr_parent = np.zeros((parent_size, ))
        self.betas_arr_parent = np.zeros((parent_size, self.p))
        self.residuals_arr_parent[0, :] = self.compute_residuals(self.beta0, self.betas)
        self.beta0_arr_parent[0] = self.beta0
        self.betas_arr_parent[0, :] = self.betas[:]
        self.num_parent = 1

        # Create beam search children. parent[i] -> child[i*child_size:(i+1)*child_size]
        total_child_size = parent_size * child_size
        self.residuals_arr_child = np.zeros((total_child_size, self.n))
        self.beta0_arr_child = np.zeros((total_child_size, ))
        self.betas_arr_child = np.zeros((total_child_size, self.p))
        self.loss_arr_child = 1e12 * np.ones((total_child_size, ))
        self.forbidden_support = set()

        num_iter = min(k, self.p)
        for i in tqdm(range(num_iter)):
            self.beamSearch_multipleSupports_via_OMP_by_1(parent_size=parent_size, child_size=child_size)

        self.residuals, self.beta0, self.betas = self.residuals_arr_parent[0], self.beta0_arr_parent[0], self.betas_arr_parent[0]
