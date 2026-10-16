import numpy as np
from pathlib import Path

from he.main import (
    ACT_FUNC, APPROX_DEGREE, APPROX_INTERVAL,
    BATCH_SIZE, LR, VERBOSITY, MOMENTUM,
)
from he.prisafltext import PriSAFLText
from he.sigmoid_he import _scaled_sigmoid_coeffs

class ModelArgs:
    def __init__(self, act_func=ACT_FUNC, approx_degree=APPROX_DEGREE,
                 approx_b=APPROX_INTERVAL, batch_size=BATCH_SIZE,
                 lr=LR, momentum=MOMENTUM, verbosity=VERBOSITY):
        self.act_func = act_func
        self.approx_degree = approx_degree
        self.approx_b = approx_b
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.verbosity = verbosity

def setup_ckks_context(ring_dim_exp, num_levels):
    from openfhe import (
        CCParamsCKKSRNS, SecretKeyDist, SecurityLevel,
        ScalingTechnique, GenCryptoContext, PKESchemeFeature,
        get_native_int,
    )

    parameters = CCParamsCKKSRNS()
    parameters.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)

    parameters.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
    parameters.SetRingDim(1 << ring_dim_exp)
    if get_native_int() == 128:
        parameters.SetScalingModSize(78)
        parameters.SetScalingTechnique(ScalingTechnique.FIXEDAUTO)
        parameters.SetFirstModSize(89)
    else:
        parameters.SetScalingModSize(59)
        parameters.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
        parameters.SetFirstModSize(60)

    parameters.SetMultiplicativeDepth(num_levels)

    from he.encryption import context_keyswitch_ok, MAX_CONTEXT_TRIES

    for attempt in range(MAX_CONTEXT_TRIES):
        cc = GenCryptoContext(parameters)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.KEYSWITCH)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)

        keys = cc.KeyGen()
        cc.EvalMultKeyGen(keys.secretKey)

        actual_ring_dim_exp = cc.GetRingDimension().bit_length() - 1
        rot_steps = []
        for i in range(actual_ring_dim_exp):
            rot_steps.append(1 << i)
            rot_steps.append(-(1 << i))
        cc.EvalRotateKeyGen(keys.secretKey, rot_steps)

        if context_keyswitch_ok(cc, keys):
            break
    else:
        raise RuntimeError(
            f"CKKS context key-switching invalid after {MAX_CONTEXT_TRIES} rebuilds")

    class CKKSCtx:
        pass
    ctx = CKKSCtx()
    ctx.cc = cc
    ctx.keys = keys
    ctx.depth = num_levels
    ctx.n_clients = 1

    return ctx

def load_pathway_mask():
    csv_path = Path(__file__).parent.parent.parent / "data" / "result.csv"
    rows = []
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            tokens = line.strip().split(",")
            rows.append([int(x) for x in tokens[2:]])
    mask = np.array(rows, dtype=np.float64)
    assert mask.shape == (37, 138), f"Unexpected mask shape: {mask.shape}"
    return mask

def plaintext_backward(model, intermediates, grad, apply_mask=True):
    n_layers = len(model.layers)
    g = grad
    batch_size = grad.shape[0]
    weight_grads = []

    for i in range(n_layers - 1, -1, -1):
        h, z_pre, a = intermediates[i]

        if not model.layers[i].last:
            act_deriv = model.activation.plain_backward(z_pre)
            g = g * act_deriv

        dW = h.T @ g / batch_size

        if apply_mask and model.layers[i].pathway_mask is not None:
            dW = dW * model.layers[i].pathway_mask

        weight_grads.append(dW)

        W = model.layers[i].W_plain[:model.layers[i].in_dim, :model.layers[i].out_dim]
        g = g @ W.T

    return list(reversed(weight_grads))

def decrypt_output(z_he, out_dim):
    decrypted = np.array(z_he._return_decrypted())
    mask = z_he.validation_mask[:len(decrypted)]
    valid = decrypted[mask == 1]
    return valid[:out_dim]

def decrypt_scalar(z_he):
    return decrypt_output(z_he, 1)[0]

def sigmoid_plain(x, interval=15.0, poly_degree=3):
    scaled_coeffs = _scaled_sigmoid_coeffs(poly_degree, interval)
    return sum(coeff * x ** power for power, coeff in scaled_coeffs.items())

def plaintext_forward(model, X):
    intermediates = []
    h = X
    for layer in model.layers:
        W = layer.W_plain[:layer.in_dim, :layer.out_dim]
        z = h @ W
        if not layer.last:
            a = model.activation.plain_forward(z)
            intermediates.append((h, z, a))
            h = a
        else:
            intermediates.append((h, z, z))
            h = z
    if h.shape[-1] == 1:
        h = h.squeeze(-1)
    return h, intermediates

def get_weights(model):
    weights = []
    for layer in model.layers:
        W = layer.W_plain[:layer.in_dim, :layer.out_dim]
        weights.append(W.copy())
    return weights

def set_weights(model, weights_list):
    for i, (layer, W_new) in enumerate(zip(model.layers, weights_list)):
        if i == 0 and layer.pathway_mask is not None:
            W_new = W_new * layer.pathway_mask
        W_padded = np.pad(W_new,
            ((0, layer.padded_in_dim - layer.in_dim),
             (0, layer.padded_out_dim - layer.out_dim)),
            mode='constant')
        layer.W_plain = W_padded
        layer.W = PriSAFLText(W_padded)
        layer.W_prev = None
        layer._x_main = None
        layer.invalidate_weight_cache()
