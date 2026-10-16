import numpy as np
from he.prisafltext import PriSAFLText
from he.layer import LayerHE
from he.activation import Activation
from he.utils import next_power_of_two

class PriSAFLHE:

    def __init__(self, dims, pathway_mask, args):
        self.dims = dims
        self.args = args
        self.batch_size = args.batch_size
        self.learning_rate = args.lr
        self.momentum = getattr(args, "momentum", 0.0)

        self.dim_slots = [next_power_of_two(d) for d in dims]

        self.matrix_sizes = [self.dim_slots[i] * self.dim_slots[i+1]
                             for i in range(len(self.dim_slots)-1)] + [self.dim_slots[-1]]
        self.slot_sizes = [max(self.matrix_sizes[i:]) for i in range(len(self.matrix_sizes))]
        self.max_slot_size = max(self.slot_sizes)
        self.slot_multipliers = [self.max_slot_size // ms for ms in self.matrix_sizes]

        print(f"dim_slots: {self.dim_slots}")
        print(f"matrix_sizes: {self.matrix_sizes}")
        print(f"max_slot_size: {self.max_slot_size}")
        print(f"slot_multipliers: {self.slot_multipliers}")

        PriSAFLText.set_verbosity(args.verbosity)

        activation = Activation(args)
        self.activation = activation

        self.layers = []
        for i in range(len(dims) - 1):
            mask = pathway_mask if i == 0 else None
            layer = LayerHE(
                rank=i + 1,
                dims=dims,
                dim_slots=self.dim_slots,
                max_slot_size=self.max_slot_size,
                slot_multipliers=self.slot_multipliers,
                activation=activation,
                lr=self.learning_rate,
                momentum=self.momentum,
                pathway_mask=mask
            )
            self.layers.append(layer)

        self.cur_data_batch_size = None

    def forward(self, X):
        n_cols = X.shape[1]
        next_pow2 = next_power_of_two(n_cols)
        pad_width = ((0, 0), (0, next_pow2 - n_cols))
        X_padded = np.pad(X, pad_width, mode='constant', constant_values=0)
        X_ht = PriSAFLText(X_padded)

        self.cur_data_batch_size = X_ht.shape[0]

        z = [X_ht]
        for i in range(len(self.layers)):
            z.append(self.layers[i].forward(z[-1], self.cur_data_batch_size))
        z = z[1:]

        return z

    def backward(self, X, g):
        z = g
        g_current = z[-1]
        gradients = []

        for i in range(len(self.layers) - 1, 0, -1):
            g_current, dW = self.layers[i].backward(z[i-1], g_current, self.cur_data_batch_size)
            gradients.append(dW)

        XT = PriSAFLText(X.T)
        XT._pad_data("col", pad_step=self.dim_slots[0], pad_batch=-1 * self.slot_sizes[0] // self.dim_slots[0])
        _, dW = self.layers[0].backward(XT, g_current, self.cur_data_batch_size)
        gradients.append(dW)

        return list(reversed(gradients))

    def apply_lookahead(self):
        for layer in self.layers:
            layer.lookahead()

    def update_weights(self, gradients, learning_rate, is_last_iteration=False, bootstrap=False):
        for layer, gradient in zip(self.layers, gradients):
            layer.update_weights(gradient, learning_rate, self.cur_data_batch_size,
                                 is_last_iteration=is_last_iteration,
                                 bootstrap=bootstrap)
