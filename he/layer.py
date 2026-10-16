import numpy as np
from he.prisafltext import PriSAFLText
from he.utils import profile, timer_context

class LayerHE():

    def __init__(self, rank, dims, dim_slots, max_slot_size, slot_multipliers, activation, lr, momentum=0.0, pathway_mask=None):
        self.rank = rank
        self.dim_slots = dim_slots
        self.max_slot_size = max_slot_size
        self.slot_multipliers = slot_multipliers
        self.activation = activation
        self.lr = PriSAFLText(lr)

        self.in_dim = dims[rank - 1]
        self.out_dim = dims[rank]
        self.padded_in_dim = dim_slots[rank - 1]
        self.padded_out_dim = dim_slots[rank]

        sm = slot_multipliers[rank - 1]
        if rank % 2 == 1:
            ps = sm * self.padded_in_dim
            self._weight_padded_shape = (max(self.padded_in_dim, ps), self.padded_out_dim)
        else:
            ps = sm * self.padded_out_dim
            self._weight_padded_shape = (self.padded_in_dim, max(self.padded_out_dim, ps))

        std = np.sqrt(2.0 / (self.in_dim + self.out_dim))
        W = np.random.randn(self.in_dim, self.out_dim) * std

        self.pathway_mask = pathway_mask
        self.padded_mask = None
        self._forward_mask_packed = None
        if pathway_mask is not None:
            W = W * pathway_mask
            self.padded_mask = np.pad(pathway_mask,
                ((0, self.padded_in_dim - self.in_dim),
                 (0, self.padded_out_dim - self.out_dim)),
                mode='constant')

        W_padded = np.pad(W, ((0, self.padded_in_dim - self.in_dim), (0, self.padded_out_dim - self.out_dim)), mode='constant')
        self.W_plain = W_padded.copy()
        self.W = PriSAFLText(W_padded)

        self.momentum = momentum
        self.W_prev = None
        self._x_main = None
        self.preZ_deriv = None

        self.last = self.rank == len(self.dim_slots) - 1

    @profile
    def forward(self, inp, cur_data_batch_size):
        PriSAFLText.verbose_print(f"{'-' * 50}Calculating z{self.rank}{'-' * 50}", 1)

        slot_multiplier = self.slot_multipliers[self.rank - 1]
        pad_step = slot_multiplier * self.padded_in_dim if self.rank % 2 == 1 else slot_multiplier * self.padded_out_dim

        if self.W._padded_data is None and self.W._cipher_data is None:
            order = "col" if self.rank % 2 == 1 else "row"
            self.W._pad_data(order, pad_step=pad_step, pad_batch=cur_data_batch_size)
            self.W.dump_info(f"W{self.rank}")

        mult_types = "PC" if self.rank == 1 else "CC"

        if self.rank == 1 and self.padded_mask is not None:
            if self._forward_mask_packed is None:
                mask_pt = PriSAFLText(self.padded_mask.astype(float))
                order_w = "col" if self.rank % 2 == 1 else "row"
                mask_pt._pad_data(order_w, pad_step=pad_step, pad_batch=cur_data_batch_size)
                self._forward_mask_packed = mask_pt._padded_data
            if inp._padded_data is None and inp._cipher_data is None:
                inp._pad_data("row", *self.W.shape)
            inp._padded_data = inp._padded_data * self._forward_mask_packed

        if self.rank % 2 == 0:
            rotate = [pad_step, self.dim_slots[self.rank - 1]]
        else:
            rotate = [1, self.dim_slots[self.rank - 1]]

        z = inp.multiply(self.W, mult_types, rotate)
        self._log_level(f"after matmul ({mult_types})", z)

        if not self.last:
            has_activation = (self.activation.args.act_func is not None
                              and self.activation.args.act_func != 'none')
            act_deriv = None

            if has_activation and self.out_dim < self.padded_out_dim:
                ones_pos = np.where(z.validation_mask == 1)[0]
                corrected = np.zeros_like(z.validation_mask)
                for c in range(0, len(ones_pos), self.padded_out_dim):
                    corrected[ones_pos[c:c + self.out_dim]] = 1
                z.validation_mask = corrected

            if has_activation:
                act_deriv = self.activation.backward(z)
                self._log_level("after act_deriv", act_deriv)

                z = self.activation.forward(z)
                self._log_level("after activation", z)

            if self.rank % 2 == 1:
                z = z._rotate_ciphertext("right", 1, self.dim_slots[self.rank+1])
                if act_deriv is not None:
                    act_deriv = act_deriv._rotate_ciphertext("right", 1, self.dim_slots[self.rank+1])
            else:
                next_n_rows = self.dim_slots[self.rank]
                next_n_cols = self.dim_slots[self.rank + 1]
                next_slot_multiplier = self.slot_multipliers[self.rank]

                if (self.rank + 1) % 2 == 1:
                    next_n_rows = next_n_rows * next_slot_multiplier
                else:
                    next_n_cols = next_n_cols * next_slot_multiplier

                z = z._rotate_ciphertext("right", next_n_rows, next_n_cols)
                if act_deriv is not None:
                    act_deriv = act_deriv._rotate_ciphertext("right", next_n_rows, next_n_cols)

            self.preZ_deriv = act_deriv
        return z

    @profile
    def backward(self, z, g, cur_data_batch_size):
        PriSAFLText.verbose_print(f"{'-' * 50}Calculating z{self.rank-1}g{'-' * 50}", 1)

        if not self.last:
            if self.activation.args.act_func is not None and self.activation.args.act_func != 'none':
                g = g.mult(self.preZ_deriv)

        if self.rank % 2 == 1:
            g = g._rotate_ciphertext("right", 1, self.dim_slots[self.rank - 1])
        else:
            g = g._rotate_ciphertext("right", self.max_slot_size // self.dim_slots[self.rank - 1], self.dim_slots[self.rank - 1])

        zg = z.mult(g, "PC" if self.rank == 1 else "CC")
        if cur_data_batch_size > 1:
            zg = zg.mult(PriSAFLText(1/cur_data_batch_size), "CD")
            zg = zg._rotate_ciphertext("left", self.max_slot_size, cur_data_batch_size)

        gW = g
        if self.rank > 1:
            if self.rank % 2 == 0:
                rotate_params = [1, self.dim_slots[self.rank]]
            else:
                rotate_params = [self.max_slot_size // self.dim_slots[self.rank], self.dim_slots[self.rank]]

            gW = g.matrix_multiply(self.W, "CC", rotate=rotate_params)

        return gW, zg

    def lookahead(self):
        self._x_main = self.W
        if self.momentum > 0.0 and self.W_prev is not None:
            mu = PriSAFLText(self.momentum)
            diff = self.W.subtract(self.W_prev, "CC")
            self.W = self.W.add(diff.mult(mu, "CD"), "CC")
            self._log_level("lookahead y", self.W)

    def update_weights(self, zg, learning_rate, cur_data_batch_size, is_last_iteration=False, bootstrap=False):
        with timer_context('UpdateWeights'):
            self.lr = PriSAFLText(learning_rate)
            update_W = zg.mult(self.lr, "CD")
            self._log_level("after lr*grad (CD)", update_W)

            if self.rank > 2:
                rotate_step = self.slot_multipliers[self.rank] * self.dim_slots[self.rank]
                update_W = update_W._rotate_ciphertext("right", rotate_step, cur_data_batch_size)
            else:
                update_W = update_W._rotate_ciphertext("right", self.dim_slots[0] * self.dim_slots[1], cur_data_batch_size)

            if self.momentum > 0.0:
                self.W_prev = self._x_main
            self.W = self.W.subtract(update_W)
            self._log_level("after W update", self.W)

        if bootstrap and not is_last_iteration:
            self.W.apply_bootstrapping_if_needed()
            self._log_level("after W bootstrap", self.W)
        self.W.shape = self._weight_padded_shape

    def _log_level(self, label, ht):
        if PriSAFLText.VERBOSE >= 1 and ht is not None:
            lvl = ht.get_level()
            if lvl is not None:
                print(f"    [level] L{self.rank} {label}: {lvl}")

    def invalidate_weight_cache(self):
        self.W._padded_data = None
        self.W._plaintext_data = None
        self.W._cipher_data = None
        self.W.shape = (self.padded_in_dim, self.padded_out_dim)
        self.W.n_rows = self.padded_in_dim
        self.W.n_cols = self.padded_out_dim
