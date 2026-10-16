import he.tests.test_pipeline as tp
import sys, time, collections, openfhe
from he.prisafltext import PriSAFLText
import he.layer as layer_mod

def nbytes(ct):
    return len(openfhe.Serialize(ct, openfhe.BINARY))

rec = collections.defaultdict(list)

_sum = PriSAFLText.sum_ciphertexts
import inspect
def sum_ciphertexts(texts):
    if inspect.stack()[1].function == 'train_epoch_he':
        for t in texts:
            rec['upload gradient'].append((nbytes(t.cipher_data), t.cipher_data.GetLevel()))
    return _sum(texts)
PriSAFLText.sum_ciphertexts = staticmethod(sum_ciphertexts)

cls = next(v for v in vars(layer_mod).values() if isinstance(v, type) and hasattr(v, 'lookahead'))
_la = cls.lookahead
def lookahead(self):
    _la(self)
    if self.W._cipher_data is not None:
        rec['broadcast look-ahead weight'].append((nbytes(self.W._cipher_data), self.W._cipher_data.GetLevel()))
cls.lookahead = lookahead

def apply_bootstrapping(self):
    ctx = self.ckks_context; cc = ctx.cc; parties = ctx.parties
    if ctx.n_clients == 1:
        raise SystemExit("K>1 only")
    ct = self._cipher_data
    rec['weight before refresh'].append((nbytes(ct), ct.GetLevel()))
    t = time.perf_counter(); inCtxt = cc.IntMPBootAdjustScale(ct); rec['t adjust (server)'].append(time.perf_counter() - t)
    t = time.perf_counter(); a = cc.IntMPBootRandomElementGen(parties[0].publicKey); rec['t random element (server)'].append(time.perf_counter() - t)
    rec['random element a'].append((nbytes(a), a.GetLevel()))
    c1 = inCtxt.Clone(); c1.RemoveElement(0)
    rec['c1 sent to each client'].append((nbytes(c1), c1.GetLevel()))
    shares = []
    for p in parties:
        t = time.perf_counter(); sp = cc.IntMPBootDecrypt(p.secretKey, c1, a); rec['t share per client'].append(time.perf_counter() - t)
        rec['share h0'].append((nbytes(sp[0]), sp[0].GetLevel())); rec['share h1'].append((nbytes(sp[1]), sp[1].GetLevel()))
        shares.append(sp)
    t = time.perf_counter(); agg = cc.IntMPBootAdd(shares); out = cc.IntMPBootEncrypt(ctx.keys.publicKey, agg, a, inCtxt)
    rec['t add+encrypt (server)'].append(time.perf_counter() - t)
    rec['refreshed weight'].append((nbytes(out), out.GetLevel()))
    self._cipher_data = out
PriSAFLText.apply_bootstrapping = apply_bootstrapping

try:
    sys.argv[0] = 'test_pipeline'; tp.main()
except SystemExit:
    pass
from he.utils import global_timer as gt
print("\n=== timers (total s / count) ===")
for c in gt.counts:
    print(f"{c:20s} {gt.timers[c]:.2f} / {gt.counts[c]}")
print("\n=== measured (MB = 1e6 bytes) ===")
for k, v in rec.items():
    if k.startswith('t '):
        print(f"{k:32s} n={len(v):3d} mean {sum(v)/len(v):.3f} s")
    else:
        sz = [b for b, _ in v]; lv = sorted({l for _, l in v})
        print(f"{k:32s} n={len(v):3d} {min(sz)/1e6:.2f}-{max(sz)/1e6:.2f} MB  levels {lv}  seq {[(round(b/1e6,2),l) for b,l in v]}")
