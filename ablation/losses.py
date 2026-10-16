import torch
import numpy as np

class ApproxCoxPHLoss(torch.nn.Module):
    def __init__(self, approx_exp_fn, approx_inv_fn):
        super().__init__()
        self.approx_exp = approx_exp_fn
        self.approx_inv = approx_inv_fn

    def forward(self, log_h, durations, events):
        idx = durations.sort(descending=True)[1]
        events_sorted = events[idx].float().view(-1)
        log_h_sorted = log_h[idx].view(-1)
        return _ApproxCoxPHFunc.apply(
            log_h_sorted, events_sorted, self.approx_exp, self.approx_inv)

class _ApproxCoxPHFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_h, events, approx_exp, approx_inv):
        if events.sum() == 0:
            raise ValueError("ApproxCoxPHLoss requires at least one event in the batch")
        ctx.save_for_backward(log_h, events)
        ctx.approx_exp = approx_exp
        ctx.approx_inv = approx_inv
        gamma = log_h.max()
        log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(1e-7).log().add(gamma)
        return -log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())

    @staticmethod
    def backward(ctx, grad_output):
        log_h, events = ctx.saved_tensors
        n_events = events.sum()
        gamma = log_h.mean() if getattr(ctx.approx_exp, 'shift', None) == 'mean' else log_h.max()
        shifted = log_h - gamma
        exp_shifted = ctx.approx_exp(shifted)
        cumsum_shifted = exp_shifted.cumsum(0)
        inv_cumsum = ctx.approx_inv(cumsum_shifted)
        weighted_inv = inv_cumsum * events
        rev_cumsum_inv = weighted_inv.flip(0).cumsum(0).flip(0)
        grad = (-events + exp_shifted * rev_cumsum_inv) / n_events
        return grad_output * grad, None, None, None

class _ApproxCoxCCPairwise(torch.autograd.Function):
    @staticmethod
    def forward(ctx, delta, approx_sigmoid_fn):
        ctx.save_for_backward(delta)
        ctx.approx_sigmoid_fn = approx_sigmoid_fn
        return torch.logaddexp(torch.zeros_like(delta), delta)

    @staticmethod
    def backward(ctx, grad_output):
        delta, = ctx.saved_tensors
        approx_grad = ctx.approx_sigmoid_fn(delta)
        return grad_output * approx_grad, None

class CoxCCLoss(torch.nn.Module):
    def __init__(self, approx_sigmoid_fn=None, n_control=1):
        super().__init__()
        self.approx_sigmoid_fn = approx_sigmoid_fn
        self.n_control = n_control

    def forward(self, risk_scores, durations, events):
        n = risk_scores.shape[0]
        risk_scores = risk_scores.squeeze()
        event_mask = (events == 1)
        event_indices = torch.where(event_mask)[0]
        if len(event_indices) == 0:
            return torch.tensor(0.0, device=risk_scores.device)

        dur_np = durations.detach().cpu().numpy()
        evt_np = event_indices.cpu().numpy()
        control_indices = []
        valid_events = []
        for i in evt_np:
            candidates = np.where((dur_np >= dur_np[i]) & (np.arange(n) != i))[0]
            if len(candidates) == 0:
                continue
            ctrl = [np.random.choice(candidates)] if self.n_control == 1 else np.random.choice(candidates, self.n_control)
            control_indices += list(ctrl)
            valid_events += [i] * len(ctrl)

        if len(valid_events) == 0:
            return torch.tensor(0.0, device=risk_scores.device)

        i_idx = torch.tensor(valid_events, device=risk_scores.device)
        j_idx = torch.tensor(control_indices, device=risk_scores.device)
        delta = risk_scores[j_idx] - risk_scores[i_idx]

        if self.approx_sigmoid_fn is not None:
            pairwise = _ApproxCoxCCPairwise.apply(delta, self.approx_sigmoid_fn)
        else:
            pairwise = torch.logaddexp(torch.zeros_like(delta), delta)
        return pairwise.sum() / (n * self.n_control)

class CoxCCPreSampledLoss(torch.nn.Module):
    def __init__(self, n_total, approx_sigmoid_fn=None):
        super().__init__()
        self.n_total = n_total
        self.approx_sigmoid_fn = approx_sigmoid_fn

    def forward(self, risk_scores, event_idx, control_idx):
        risk_scores = risk_scores.squeeze()
        delta = risk_scores[control_idx] - risk_scores[event_idx]
        if self.approx_sigmoid_fn is not None:
            pairwise = _ApproxCoxCCPairwise.apply(delta, self.approx_sigmoid_fn)
        else:
            pairwise = torch.logaddexp(torch.zeros_like(delta), delta)
        return pairwise.sum() / self.n_total

class CoxCCMiniBatchLoss(torch.nn.Module):
    def __init__(self, n_events=1):
        super().__init__()
        self.n_events = n_events

    def forward(self, risk_scores):
        risk_scores = risk_scores.squeeze()
        h_events = risk_scores[:self.n_events]
        h_controls = risk_scores[self.n_events:]
        loss = torch.tensor(0.0, device=risk_scores.device)
        for h_e in h_events:
            delta = h_controls - h_e
            loss = loss + torch.logaddexp(torch.zeros_like(delta), delta).mean()
        return loss / self.n_events
