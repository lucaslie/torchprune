"""Module containing the allocators for SiPP pruning."""
import copy

import numpy as np
import cvxpy as cp
import torch

from ..base import BaseAllocator


class SiPPRandAllocator(BaseAllocator):
    """The layer allocator for randomized SiPP."""

    def __init__(self, trackers, delta_failure, c_constant, **kwargs):
        """Initialize w/ sens tracker and constants from the theorems."""
        super().__init__(**kwargs)
        self.register_buffer("_coeffs", torch.Tensor())
        self.register_buffer("_l_coeffs", torch.Tensor())
        self.register_buffer("_nnz", torch.Tensor())
        self.register_buffer("_sum_sens", torch.Tensor())
        self.register_buffer("_valid", torch.Tensor())
        self.register_buffer("_eps", torch.Tensor())
        self._probability = {}
        self._delta_failure = delta_failure
        self._c_constant = c_constant
        self._num_layers = len(trackers)

        # initialize some stuellf properly
        device = trackers[0].sensitivity.device
        self._eps = torch.Tensor([np.finfo(np.float32).eps]).to(device)

        # do initialization per layer
        for ell, tracker in enumerate(trackers):
            # compute relevant quantities
            coeffs, nnz, l_coeffs, sum_sens, probs = self._get_coefficients(
                tracker
            )

            # store offset
            num_el = self._coeffs.numel()  # pylint: disable=E0203
            self._offsets[ell] = [num_el, num_el + coeffs.numel()]

            # store coefficients in flat array
            if self._coeffs.numel():  # pylint: disable=E0203
                self._coeffs = torch.cat((self._coeffs, coeffs))
                self._l_coeffs = torch.cat((self._l_coeffs, l_coeffs))
                self._nnz = torch.cat((self._nnz, nnz))
                self._sum_sens = torch.cat((self._sum_sens, sum_sens))
            else:
                self._coeffs = coeffs
                self._l_coeffs = l_coeffs
                self._nnz = nnz
                self._sum_sens = sum_sens

            self._probability[ell] = probs

        # remember the filters that should be sampled
        self._valid = self._sum_sens > self._eps

    def _get_coefficients(self, tracker):
        """Get the coefficients according to our theorems."""
        num_filters = tracker.sensitivity.shape[0]
        num_patches = tracker.num_patches

        # a few stats from sensitivity
        nnz, sum_sens, probs = self._get_sens_stats(tracker)

        # cool stuff
        k_size = tracker.module.weight[0, 0].numel()
        log_numerator = torch.tensor(8.0 * (num_patches + 1) * k_size).to(
            nnz.device
        )
        log_term = self._c_constant * torch.log(
            log_numerator / self._delta_failure
        )
        alpha = 2.0 * log_term

        # compute coefficients
        coeffs = copy.deepcopy(sum_sens)
        coeffs *= alpha
        # compute leading coefficients
        l_coeffs = torch.ones_like(coeffs)
        l_coeffs = self._adapt_l_coeffs(l_coeffs, num_filters, num_patches)

        return coeffs, nnz, l_coeffs, sum_sens, probs

    def _adapt_l_coeffs(self, l_coeffs, num_filters, num_patches):
        l_coeffs /= num_filters
        return l_coeffs

    def _get_sens_stats(self, tracker):
        # short-hand
        sens = tracker.sensitivity

        # function for one side
        def _get_one_side(idx):
            sens_pm = copy.deepcopy(sens)
            sens_pm[~idx] = 0.0
            nnz_pm = (sens_pm != 0.0).view(sens_pm.shape[0], -1).sum(dim=-1)

            sum_sens_pm = sens_pm.view(sens_pm.shape[0], -1).sum(dim=-1)
            probs = sens_pm.view(sens_pm.shape[0], -1) / sum_sens_pm.unsqueeze(
                -1
            )
            return nnz_pm, sum_sens_pm, probs

        nnz_p, sum_sens_p, probs_p = _get_one_side(tracker.idx_plus)
        nnz_m, sum_sens_m, probs_m = _get_one_side(tracker.idx_minus)

        # process stats
        nnz = torch.cat((nnz_p, nnz_m))
        sum_sens = torch.cat((sum_sens_p, sum_sens_m))
        probs = torch.cat((probs_p, probs_m))

        return nnz, sum_sens, probs

    def _get_f_error(self, idx):
        # get relevant coefficients for desired index set
        coeffs = self._coeffs[idx].cpu().numpy()
        coeffs[coeffs < 0.0] = 0.0

        # define variables and parameters
        num_var = coeffs.shape[0]
        x_arg = cp.Variable(num_var)
        alpha = cp.Parameter(num_var, nonneg=True)
        alpha.value = coeffs

        # construct symbolic error vector for theoretical error per filter
        k_constant = 3
        expr = cp.vstack(
            [
                cp.multiply(cp.inv_pos(x_arg), alpha / k_constant),
                cp.multiply(
                    cp.inv_pos(cp.sqrt(x_arg)), cp.sqrt(6 * alpha / k_constant)
                ),
            ]
        )
        f_error = cp.norm(expr, axis=0) + cp.multiply(
            cp.inv_pos(x_arg), alpha / k_constant
        )
        f_error = 1 / 2 * f_error

        # return argument and symbolic error function to argument
        return x_arg, f_error

    def _allocate_method(self, budget):
        # obtain symbolic error vector
        x_arg, f_error = self._get_f_error(self._valid)

        # Declare constraints
        # The last constraint x <= nnzRow is to ensure that we don't sample
        # more than the number of nonzero entries of the matrix.
        # x >= 0 is the correct constraint. Otherwise, we get unintended
        # optimal results
        nnz_prob_per_filter = self._nnz[self._valid].cpu().numpy()
        constraints = [
            x_arg >= 0,
            x_arg <= nnz_prob_per_filter,
            self._budget_constraint(x_arg, budget),
        ]

        # set objective
        l_coeffs = self._l_coeffs[self._valid].cpu().numpy()
        f_objective = cp.Minimize(cp.sum(cp.multiply(l_coeffs, f_error)))

        # solve
        prob = cp.Problem(f_objective, constraints)

        try:
            prob.solve(solver=cp.MOSEK)
        except cp.error.SolverError:
            prob.solve()

        # store optimal allocation
        self._allocation = torch.zeros_like(self._coeffs, dtype=torch.int)
        x_arg = (
            torch.Tensor(np.ceil(x_arg.value))
            .int()
            .to(self._allocation.device)
        )
        self._allocation[self._valid] = x_arg

    def _budget_constraint(self, x_var, m_budget):
        return cp.sum(x_var) - m_budget <= 0

    def _get_error_theoretical(self, ell, num_samples):
        # get symbolic error function
        idx = range(self._offsets[ell][0], self._offsets[ell][1])
        x_var, error = self._get_f_error(idx)

        # assign value to variables (and do some sanity check on num_samples)
        num_samples_np = num_samples.cpu().view(-1).numpy()
        num_samples_np[
            np.logical_or(num_samples_np < 1, ~np.isfinite(num_samples_np))
        ] = 1
        x_var.project_and_assign(num_samples_np)

        # compute error
        error = error.value
        error = torch.Tensor(error).to(num_samples.device).view_as(num_samples)

        return error


class SiPPAllocator(BaseAllocator):
    """Allocator for deterministic, sensitivity-based weight pruning."""

    def __init__(self, trackers, delta_failure=None, C=1, **kwargs):
        """Initialize the alloctor with signature consistent with others."""
        super().__init__(**kwargs)
        self.register_buffer("_sensitivity", torch.Tensor())
        self._num_layers = len(trackers)
        self._c_constant = C
        self.register_buffer("_num_filters", torch.zeros(self._num_layers))
        self._offset_sensitivity = {}

        # for easier bookkeeping
        self._offsets[-1] = [0, 0]

        for ell, tracker in enumerate(trackers):
            sens_ell, self._num_filters[ell], offset = self._get_sens_stats(
                tracker
            )
            num_el = self._sensitivity.numel()  # pylint: disable=E0203
            self._offset_sensitivity[ell] = [num_el, num_el + sens_ell.numel()]

            self._offsets[ell] = [
                self._offsets[ell - 1][1],
                self._offsets[ell - 1][1] + offset,
            ]
            if self._sensitivity.numel():  # pylint: disable=E0203
                self._sensitivity = torch.cat((self._sensitivity, sens_ell))
            else:
                self._sensitivity = sens_ell

    def _get_sens_stats(self, tracker):
        sens_p = self._get_sens_pm(tracker.sensitivity, tracker.idx_plus)
        sens_m = self._get_sens_pm(tracker.sensitivity, tracker.idx_minus)
        sens = torch.cat((sens_p.view(-1), sens_m.view(-1)))
        num_filters = sens_p.shape[0]
        offset = num_filters * 2
        return sens, num_filters, offset

    def _get_sens_pm(self, sens, idx):
        sens_pm = copy.deepcopy(sens)
        sens_pm[~idx] = 0.0
        return sens_pm

    def _get_top_indicator(self, arr, num_samples):
        # pre-allocate indicators
        indicator = torch.zeros_like(arr, dtype=torch.bool)

        # get top indices and assign true in indicator
        _, idx_top = torch.topk(arr.view(-1), int(num_samples), sorted=False)
        indicator.view(-1)[idx_top] = True

        return indicator

    def _allocate_method(self, budget):
        # throw out old stuff
        self._allocation = torch.Tensor()

        indicator = self._get_top_indicator(self._sensitivity, budget)

        for ell in range(self._num_layers):
            indicator_ell = self._extract(
                indicator, self._offset_sensitivity, ell
            )
            allocation_ell = self._allocate_samples_one(
                ell, indicator_ell.sum()
            ).view(-1)
            if self._allocation.numel():
                self._allocation = torch.cat(
                    (self._allocation, allocation_ell)
                )
            else:
                self._allocation = allocation_ell

    def _allocate_samples_one(self, layer, budget):
        sens_ell = self._extract(
            self._sensitivity, self._offset_sensitivity, layer
        )
        sens_ell = sens_ell.view(2, int(self._num_filters[layer]), -1)
        indicator_ell = self._get_top_indicator(sens_ell, budget)
        allocation = torch.sum(indicator_ell, dim=-1).int()
        return allocation

    def _get_error_theoretical(self, ell, num_samples):
        # extract sensitivity for this layer and reshape as
        # [2, numFilters, -1]
        sens_l = self._extract(
            self._sensitivity, self._offset_sensitivity, ell
        )

        sens_l = sens_l.view(2, self._num_filters[ell].int(), -1)

        # preallocate indicators
        indicator_l = torch.zeros_like(sens_l, dtype=torch.bool)

        # extract indicators for top sensitivity
        for idx_filter in range(num_samples.shape[1]):
            for i in range(num_samples.shape[0]):
                indicator_l[i, idx_filter] = self._get_top_indicator(
                    sens_l[i, idx_filter], num_samples[i, idx_filter]
                )

        # compute error by looking at sens values which were not picked
        sens_anti = copy.deepcopy(sens_l)
        sens_anti.masked_fill_(indicator_l, 0.0)
        error = self._c_constant * torch.sum(sens_anti, dim=-1)

        return error


class SiPPHybridAllocator(BaseAllocator):
    """This is the alloctor for hybrid SiPP."""

    def __init__(self, trackers, delta_failure, c_constant, **kwargs):
        """Initialize the alloctor consistent with the others."""
        super().__init__(**kwargs)
        self._allocator_rand = SiPPRandAllocator(
            trackers, delta_failure, c_constant
        )
        self._allocator_det = SiPPAllocator(
            trackers, delta_failure, c_constant
        )

    def _allocate_method(self, budget):
        self._allocator_rand._allocate_method(budget)

    def get_num_samples(self, layer):
        """Get the number of samples for a particular layer index."""
        # get optimal sample numbers from randAllocator
        num_samples = self._allocator_rand.get_num_samples(layer)
        no_sampling = num_samples < 0
        num_samples[no_sampling] = 1

        # check error for both methods
        error_rand = self._allocator_rand._get_error_theoretical(
            layer, num_samples
        )
        error_det = self._allocator_det._get_error_theoretical(
            layer, num_samples
        )

        # do random when random is better
        use_rand = error_det > error_rand
        num_samples[use_rand] = -num_samples[use_rand]

        # reset zero samples to zero
        num_samples[no_sampling] = 0

        return num_samples
