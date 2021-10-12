"""Module containing the ALDS allocators."""

import numpy as np
import torch

from ..base_decompose import BaseDecomposeAllocator, FoldScheme


class ALDSErrorAllocator(BaseDecomposeAllocator):
    """Relative error-based (singular values) ALDS allocator.

    We minimize the maximum relative error based on the operator
    norm, where the relative error is defined as:

    rel_error = ||W - What||_op // ||W||_op

    Note that || . ||_op is the largest singular value, i.e., l-infty norm of
    singular values.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and compute operator norm as well."""
        super().__init__(*args, **kwargs)

        # register the buffers for the required errors
        self.register_buffer("_rel_error", None)

        # compute relative error for each layer
        rel_errors = [
            self._compute_rel_error_for_weight(mod.weight, k_s, scheme)
            for mod, k_s, scheme in zip(
                self._net.compressible_layers, self._k_splits, self._schemes
            )
        ]

        # pre-allocate big tensor
        num_sv = [len(rel_e) for rel_e in rel_errors]
        self._rel_error = torch.zeros(
            len(num_sv),
            max(num_sv),
            device=rel_errors[0].device,
            dtype=torch.double,
        )

        # put errors into one big tenor
        for r_e_t, r_e in zip(self._rel_error, rel_errors):
            r_e_t[: len(r_e)].copy_(r_e)

    @staticmethod
    def _compute_sv_for_weight(weight, k_split, scheme):
        """Compute SVD for one layer."""
        # fold into matrix operator
        weight = scheme.fold(weight.detach())

        # get k_split as int instead of tensor
        k_split = k_split.item()

        # compute rank
        rank_k = min(weight.shape[0], weight.shape[1] // k_split)
        # pre-allocate singular values ...
        singular_values = torch.zeros(k_split, rank_k, device=weight.device)
        # compute singular values for each part of the decomposed weight
        for idx_k, w_per_g_k in enumerate(torch.chunk(weight, k_split, dim=1)):
            singular_values[idx_k] = torch.svd(w_per_g_k, compute_uv=False)[1]

        # note has shape [k_split x rank_per_k]
        return singular_values

    @staticmethod
    def _compute_norm_for_weight(weight, scheme, ord):
        """Compute and return operator norm for given matrix."""
        # fold into matrix operator
        weight = scheme.fold(weight.detach())

        # get and return operator norm of flattened weight
        return torch.linalg.norm(weight, ord=ord)

    def _compute_rel_error_for_weight(self, weight, k_split, scheme):
        # grab all singular values for decomposition
        # has shape [k_split x rank_per_k]
        singular_values = self._compute_sv_for_weight(weight, k_split, scheme)

        # compute operator norm for current operator
        # has shape []
        op_norm = self._compute_norm_for_weight(weight, scheme, ord=2)

        # compute operator norm of "residual operator" W - What
        # --> corresponds to biggest singular values not included.
        # What consists of multiple k_splits.
        # see corresponding paper for op_norm derivation
        # We take max over k-splits here.
        op_norm_residual = singular_values.max(dim=0)[0]

        # resulting relative error for each layer and each possible rank_j!
        # shape is [max(num_sv)]
        rel_error = op_norm_residual / op_norm

        return rel_error

    def _get_boundaries(self):
        # use min and max relative error ...
        return self._rel_error.min(), self._rel_error.max() * 1.01

    def _compute_ranks_j_for_arg(self, arg, ranks, num_weights_per_j):
        # rank_j's correspond to index of 1st relative error smaller than arg.
        # now replace rank_j's with a more refined computation.
        bigger = self._rel_error > arg

        # get ranks now per layer
        ranks_j = bigger.sum(dim=-1)

        return ranks_j


class ALDSErrorAllocatorScheme0(ALDSErrorAllocator):
    """Relative error-based ALDS allocator with scheme 0."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_ENCODE.value


class ALDSErrorAllocatorScheme1(ALDSErrorAllocator):
    """Relative error-based ALDS allocator with scheme 1."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_SPLIT1.value


class ALDSErrorAllocatorScheme2(ALDSErrorAllocator):
    """Relative error-based ALDS allocator with scheme 2."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_SPLIT2.value


class ALDSErrorAllocatorScheme3(ALDSErrorAllocator):
    """Relative error-based ALDS allocator with scheme 3."""

    @property
    def _folding_scheme_value(self):
        return FoldScheme.KERNEL_DECODE.value


class ALDSErrorIterativeAllocator(ALDSErrorAllocator):
    """Iterative optimization wrapper for rel error allocator."""

    @property
    def _num_seeds(self):
        """Get number of seeds to try for k."""
        return 15

    @property
    def _num_iter(self):
        """Get number of iterations per seed."""
        return 4

    @property
    def _fixed_k_seeds(self):
        """Get the fixed desired values of k to use as seeds."""
        return [3, 5, 7, 9]

    @property
    def _scheme_choices(self):
        """Get an iterator over scheme choices.

        Note that we assume that the index of each choice corresponds to the
        enum value.
        """
        return [FoldScheme(0)]

    def __init__(self, *args, **kwargs):
        """Initialize with the desired number of seeds and iterations."""
        super().__init__(*args, **kwargs)

        # now also store the potential values of k for each layer
        self._possible_k_splits = [
            self._get_possible_k(mod.weight.shape[1])
            for mod in self._net.compressible_layers
        ]

        # store default rounding factor
        self._rounding_factor = None
        self._reset_round_rank()

        # recall some stats
        num_layers = len(self._possible_k_splits)
        num_max_k = max(len(possible) for possible in self._possible_k_splits)
        num_schemes = len(self._scheme_choices)

        # remember current k_splits and schemes
        k_split_backup = self._k_splits.detach().clone()
        scheme_values_backup = self._scheme_values.detach().clone()

        # we need ranks per-layer per-scheme and then take the max
        self._k_splits[:] = 1
        rank_max = torch.zeros_like(self._out_features)
        for scheme in self._scheme_choices:
            self._scheme_values = scheme.value
            rank_max = torch.maximum(rank_max, self._get_weight_stats()[0])
        rank_max = torch.max(rank_max).item()

        # now reset k_splits and scheme_values
        self._k_splits = k_split_backup
        self._scheme_values = scheme_values_backup

        # pre-allocate a look-up table for rel-errors
        self._rel_error_lookup = torch.zeros(
            num_layers,
            num_max_k,
            num_schemes,
            rank_max,
            dtype=self._rel_error.dtype,
            device=self._rel_error.device,
        )

        # copy current content into lookup.
        for ell, (k_i, scheme, r_e) in enumerate(
            zip(self._k_splits, self._schemes, self._rel_error)
        ):
            lookup = self._rel_error_lookup[
                ell, self._get_k_index(ell, k_i), scheme.value
            ]
            lookup[: len(r_e)].copy_(r_e)

        # make sure that self._rel_error has right shape and update content
        self._rel_error = torch.zeros_like(self._rel_error_lookup[:, 0, 0, :])
        self._update_rel_error()

    def _lookup_rel_error(self, ell, k_split, scheme):
        """Look up desired rel-error and compute if necessary."""
        weight = self._net.compressible_layers[ell].weight
        idx_k = self._get_k_index(ell, k_split)
        lookup = self._rel_error_lookup[ell, idx_k, scheme.value]
        if not torch.any(lookup != 0.0):
            rel_error_new = self._compute_rel_error_for_weight(
                weight, k_split, scheme
            )
            lookup[: len(rel_error_new)].copy_(rel_error_new)
        return lookup

    def _update_rel_error(self):
        """Update self._rel_error by using look-up if possible."""
        for ell, k_split in enumerate(self._k_splits):
            lookup = self._lookup_rel_error(ell, k_split, self._scheme(ell))
            self._rel_error[ell].copy_(lookup)

    def _get_k_index(self, ell, k_split):
        """Get index of current k in the list of possible k's."""
        if isinstance(k_split, torch.Tensor):
            k_split = k_split.item()
        return np.where(self._possible_k_splits[ell] == k_split)[0][0]

    def _get_k_s_seed(self, i):
        """Get ith seed for values of k and schemes."""
        if i < len(self._fixed_k_seeds):
            # seed is based on some fixed k_seed and fixed scheme_choice
            self._desired_k_split = self._fixed_k_seeds[i]
            s_values = torch.zeros_like(self._scheme_values)
            s_values[:] = self._folding_scheme_value
            return self._get_k_splits(self._desired_k_split), s_values
        else:
            # randomly allocate value of k and s
            k_random = torch.zeros_like(self._k_splits)
            s_random = torch.zeros_like(self._scheme_values)
            for ell, k_options in enumerate(self._possible_k_splits):
                k_random[ell] = np.random.choice(k_options)
                s_random[ell] = np.random.choice(len(self._scheme_choices))
            return k_random, s_random

    def _reset_round_rank(self):
        """Reset the rounding rank to default."""
        self._rounding_factor = 0.5

    def _lower_round_rank(self):
        """Lower the rounding rank."""
        self._rounding_factor -= 0.2

        if self._rounding_factor < -0.51:
            error_msg = "Rounding factor too low now."
            print(error_msg)
            raise ValueError(error_msg)

    def _round_rank(self, rank):
        """Round the rank dynamically based on self._rounding_factor.

        We need this function in case we round up **too much** and overshoot
        the budget in which case the allocation becomes invalid.
        """
        rounded = torch.floor(rank + self._rounding_factor)
        rounded[rounded < 0] = 0
        return rounded

    def _find_best_k_s(self, ell):
        """Find best k for a given budget in the desired layer."""
        # some layer stats
        def _get_stats(k_split, scheme):
            kernel = self._kernel_shapes[ell]
            k_out, k_in = scheme.get_decomposed_kernel_sizes(kernel)
            out_f = self._out_features[ell] * k_out
            in_f = self._in_features[ell] * k_in
            rank = torch.min(out_f, in_f // k_split)
            num_w_per_j = k_split * out_f + in_f
            return rank, num_w_per_j

        # get budget for this layer.
        rank_original, k_original, s_original = self.get_num_samples(ell)
        scheme_original = FoldScheme(s_original.item())
        budget = rank_original * _get_stats(k_original, scheme_original)[1]

        # go through all possible values of k,s and record best value of k,s
        rel_error_best = float("Inf")
        k_best = -1
        s_best = -1

        for k_option in self._possible_k_splits[ell]:
            for scheme in self._scheme_choices:
                # check resulting relative error
                rel_error = self._lookup_rel_error(ell, k_option, scheme)

                # record stats
                rank_full, num_w_per_j = _get_stats(k_option, scheme)

                # get potential rank based on budget and weights per j
                rank_closest = self._round_rank(budget / num_w_per_j)
                if rank_closest < 1:
                    continue
                else:
                    rank_closest = min(rank_full, rank_closest)
                idx_rank = int(rank_closest - 1)

                # resulting relative error
                rel_error_resulting = rel_error[idx_rank]

                # check if resulting error is lower than best recorded
                if rel_error_resulting < rel_error_best:
                    rel_error_best = rel_error_resulting
                    k_best = k_option
                    s_best = scheme.value

        if k_best < 0:
            error_msg = f"No valid k found for layer {ell}"
            print(error_msg)
            raise ValueError(error_msg)

        # now return best found k, s
        return k_best, s_best

    def _super_allocate(self, budget, disp=True):
        """Allocate with super and raise ValueError if disp==True."""
        # update rel error according to current k splits.
        self._update_rel_error()
        super()._allocate_method(budget, disp=disp)
        return self._arg_opt

    def _allocate_method(self, budget):
        """Allocate with optimizer.

        Note that the optimizer fails if we cannot produce a valid allocation.
        Then we should modify the keep ratio to get a valid allocation...
        """
        # if allocation fails we try to adapt keep ratio to be within range.
        try:
            return self._optimize_allocation(budget)
        except ValueError:
            print("Adapting keep ratio since no valid allocation possible")

            # set k==1, s==0 ... (vanilla allocation...)
            self._k_splits.fill_(1)
            self._scheme_values = 0

            # try allocating without raising error to see what we get.
            arg_opt = self._super_allocate(budget, False)

            # compute possible budget accordingly
            budget_possible = self._get_resulting_size(arg_opt)

            # add some "wiggle" room for the budget (+/-0.5 %)
            if budget > budget_possible:
                budget_possible = (0.995 * budget_possible).to(budget_possible)
            else:
                budget_possible = (1.005 * budget_possible).to(budget_possible)

        # now try again with the possible budget
        try:
            return self._optimize_allocation(budget_possible)
        except ValueError:
            # if nothing helped we will give up and raise on
            error_msg = "Still no valid allocation after adapting budget."
            print(error_msg)
            raise ValueError(error_msg)

    def _optimize_allocation(self, budget):
        """Wrap allocator in EM-style allocator here."""

        def _super_allocate():
            """Allocate and raise ValueError if allocation fails."""
            return self._super_allocate(budget)

        def _initialize_seed():
            """Initialize seed with valid allocation.

            We need to have a valid seed to start with , i.e., a seed that
            generates a valid budget.
            Otherwise, we can try to slowly reduce the value of k.
            """
            # reset rounding rank function
            self._reset_round_rank()
            # now find a valid seed
            while True:
                try:
                    # if everything works we can break
                    rel_error_original = _super_allocate()
                    return rel_error_original
                except ValueError:
                    # halve the values of k from the split.
                    k_proposal = self._k_splits // 2

                    # in this case we had k == 1 in the previous iteration but
                    # it is still not enough to get a valid allocation.
                    if torch.all(k_proposal < 1):
                        error_msg = "No more parameter reduction achievable."
                        print(error_msg)
                        raise ValueError(error_msg)

                    # now set new proposal
                    self._k_splits.copy_(self._get_k_splits(k_proposal))
                    print("Reducing k seed to get valid allocation.")

        def _iterate():
            """Update the seed with finding the best k and re-allocate.

            However, for very small budgets we might overshoot the budget with
            the best k due to rounding issues. Thus the allocation
            becomes invalid. We should thus reduce the rounding threshold and
            repeat.
            """
            # keep a copy of the current split and rel error around
            k_splits_backup = self._k_splits.detach().clone()
            s_values_backup = self._scheme_values.detach().clone()

            # now keep trying until we have success
            while True:
                # check if we can get better performance by re-assigning k
                k_improved = torch.zeros_like(self._k_splits)
                s_improved = torch.zeros_like(self._scheme_values)
                for ell in range(self._num_layers):
                    k_improved[ell], s_improved[ell] = self._find_best_k_s(ell)
                self._k_splits.copy_(k_improved)
                self._scheme_values = s_improved

                # solve new allocation except when budget fails.
                # then lower the rounding factor for the rank and repeat
                try:
                    rel_error = _super_allocate()
                    return rel_error
                except ValueError:
                    # lower rounding factor
                    self._lower_round_rank()
                    # reset k,s and re-allocate based on k,s we started with.
                    self._k_splits.copy_(k_splits_backup)
                    self._scheme_values = s_values_backup
                    _super_allocate()
                    print("Lowering rounding factor to get valid allocation.")

        def _optimize_seed(i):
            """Optmize seed i."""
            # get the next proposed seed for values of k.
            k_seed, s_seed = self._get_k_s_seed(i)
            self._k_splits.copy_(k_seed)
            self._scheme_values = s_seed

            # try initializing seed with valid allocation now
            # and adapt seed if necessary
            rel_error_original = _initialize_seed()

            # store resulting seeds after initialization (might change)
            k_seed = self._k_splits.detach().clone()
            s_seed = self._scheme_values.detach().clone()
            k_current = k_seed.detach().clone()
            s_current = s_seed.detach().clone()

            # starting message
            print(f"Started seed {i} with relative error {rel_error_original}")
            rel_error = rel_error_original

            # iterate and update k & j values
            for j in range(self._num_iter):
                try:
                    rel_error = _iterate()
                    k_current.copy_(self._k_splits)
                    s_current.copy_(self._scheme_values)
                    print(
                        f"Seed {i}, it {j}: current relative error {rel_error}"
                    )
                except ValueError:
                    print(
                        "Failed iterating on current values of k."
                        " Stopping and keeping current values of k."
                    )
                    self._k_splits.copy_(k_current)
                    self._scheme_values = s_current
                    break

            # in case of rounding errors we can have that the updates don't
            # produce improved results...
            # It should only happen for very small prune ratios though.
            if rel_error_original < rel_error:
                print(
                    "Iterative optimization unexpectedly increased rel"
                    f"ative error from {rel_error_original} to {rel_error}"
                    " likely due to rounding issues. Keeping initial k-seed."
                )
                rel_error = rel_error_original
                self._k_splits.copy_(k_seed)
                self._scheme_values = s_seed
            else:
                # print some statistics about the overall improvement
                print(
                    f"Seed {i}: improved from {rel_error_original}"
                    f" to {rel_error}"
                )

            return rel_error

        # keeping track of over-all rel_error
        error_best = float("Inf")
        i_best = -1
        k_splits_best = torch.zeros_like(self._k_splits)
        s_values_best = torch.zeros_like(self._scheme_values)

        for i in range(self._num_seeds):
            # try optimizing for the current seed i
            try:
                rel_error = _optimize_seed(i)
            except ValueError:
                print(f"Couldn't optimize seed {i}; moving on to next seed.")
                continue

            # see if it improves best rel error from before
            if rel_error < error_best:
                error_best = rel_error
                i_best = i
                k_splits_best.copy_(self._k_splits)
                s_values_best.copy_(self._scheme_values)

        if i_best < 0:
            error_msg = "No seed produced a valid allocation."
            raise ValueError(error_msg)
        else:
            # print some final statistics.
            print(f"Best seed: {i_best}, best error: {error_best}")

        # now set best k splits and solve for the final time
        self._k_splits.copy_(k_splits_best)
        self._scheme_values = s_values_best
        _super_allocate()


class ALDSErrorIterativeAllocatorPlus(ALDSErrorIterativeAllocator):
    """Iterative optimization wrapper for rel error allocator with schemes."""

    @property
    def _num_seeds(self):
        """Get number of seeds to try for k."""
        return 30

    @property
    def _scheme_choices(self):
        """Get an iterator over scheme choices.

        Note that we assume that the index of each choice corresponds to the
        enum value.
        """
        return FoldScheme


class ALDSErrorKOnlyAllocator(ALDSErrorIterativeAllocator):
    """A constant per-layer prune ratio combined with the best k.

    This allocator will pick a constant per-layer prune ratio and then try to
    find the best k for each layer.
    """

    def __init__(self, *args, **kwargs):
        """Initialize and keep a fake rel eror around."""
        super().__init__(*args, **kwargs)
        self._rel_error_fake = 100.0

    @property
    def _num_seeds(self):
        """Get number of seeds to try for k."""
        return 1

    @property
    def _num_iter(self):
        """Get number of iterations per seed."""
        return 1

    def _get_boundaries(self):
        # use min/max relative prune ratio
        return 1e-12, 1.1

    def _compute_ranks_j_for_arg(self, arg, ranks, num_weights_per_j):
        """Return ranks such that per-layer prune ratio is constant."""
        # total size and per-layer budget
        num_w_orig = self._out_features * self._in_features * self._kernel_size
        budget_per_layer = arg * num_w_orig

        # ranks are chosen accordingly
        ranks_j = (budget_per_layer / num_weights_per_j).round()
        return ranks_j

    def _super_allocate(self, budget, disp=True):
        # Run allocate but then catch wrong arg_opt and replace it
        super()._super_allocate(budget, disp=disp)

        # since we never want to use the initialization we need to fake a lower
        # error always (just a ever decreasing fake error)
        self._rel_error_fake -= 0.01
        return self._rel_error_fake
