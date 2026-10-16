import numpy as np
import time
from collections import defaultdict

try:
    from line_profiler import profile
except ImportError:
    def profile(func):
        return func

class GlobalTimer:
    def __init__(self):
        self.timers = defaultdict(float)
        self.counts = defaultdict(int)
        self.stack = []
        self.active_timers = {}
        self.n_clients = 1
        self.num_samples = 0
        self.batch_size = 1
        self.total_iterations = 1

    def set_n_clients(self, n_clients):
        self.n_clients = n_clients

    def set_batch_info(self, num_samples, batch_size):
        self.num_samples = num_samples
        self.batch_size = batch_size

    def set_total_iterations(self, total_iterations):
        self.total_iterations = total_iterations

    def start_timer(self, category):
        start_time = time.time()
        if category in self.active_timers:
            self.stack.append((category, self.active_timers[category]))
        self.active_timers[category] = start_time

    def end_timer(self, category):
        if category not in self.active_timers:
            return

        end_time = time.time()
        elapsed = end_time - self.active_timers[category]
        self.timers[category] += elapsed
        self.counts[category] += 1

        if self.stack:
            prev_category, prev_start = self.stack.pop()
            if prev_category == category:
                self.active_timers[category] = prev_start
            else:
                self.stack.append((prev_category, prev_start))
                del self.active_timers[category]
        else:
            del self.active_timers[category]

    def print_timing_table(self):
        preferred_order = ['Forward', 'CoxCCGradient', 'Backward', 'UpdateWeights',
                           'Bootstrap', 'Aggregation']
        cats = [c for c in preferred_order if self.counts[c] > 0]
        cats += [c for c in sorted(self.timers) if c not in cats and self.counts[c] > 0]

        n_iter = max(self.total_iterations, 1)
        n_cli = max(self.n_clients, 1)
        n_samp = max(self.num_samples, 1)

        print(f"\n{'='*90}")
        print("TIMING BREAKDOWN")
        print(f"  Epochs: {n_iter}, Clients: {n_cli}, Samples: {n_samp}")
        print(f"{'='*90}")
        print(f"{'Phase':<20} {'Total (s)':>10} {'Per-Iter (s)':>12} {'Per-Sample (s)':>14} {'Calls':>6}")
        print(f"{'-'*90}")

        grand_total = 0.0
        for cat in cats:
            t = self.timers[cat]
            grand_total += t
            per_iter = t / n_iter
            per_sample = t / (n_iter * n_samp)
            print(f"  {cat:<18} {t:>10.2f} {per_iter:>12.2f} {per_sample:>14.2f} {self.counts[cat]:>6}")

        print(f"{'-'*90}")
        per_iter_total = grand_total / n_iter
        per_sample_total = grand_total / (n_iter * n_samp)
        print(f"  {'TOTAL':<18} {grand_total:>10.2f} {per_iter_total:>12.2f} {per_sample_total:>14.2f}")
        print(f"{'='*90}")

global_timer = GlobalTimer()

def timer_context(category):
    class TimerContext:
        def __init__(self, cat):
            self.category = cat

        def __enter__(self):
            global_timer.start_timer(self.category)
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            global_timer.end_timer(self.category)

    return TimerContext(category)

def is_power_of_two(n):
    return n > 0 and (n & (n - 1)) == 0

def next_power_of_two(n):
    if is_power_of_two(n):
        return n
    return 2**int(np.ceil(np.log2(n)))

def distance_to_next_power_of_two(n):
    if is_power_of_two(n):
        return 0
    return next_power_of_two(n) - n
