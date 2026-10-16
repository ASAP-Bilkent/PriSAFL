#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")"

pass=0; fail=0
ok()   { echo -e ">> PASS: $*"; pass=$((pass+1)); }
bad()  { echo -e ">> FAIL: $*"; fail=$((fail+1)); }

echo "============================================================"
echo " 1/4  imports + environment"
echo "============================================================"
./run.sh 'python - <<PY
import numpy, openfhe, torch
print("numpy   ", numpy.__version__)
print("torch   ", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "| devs:", torch.cuda.device_count())
print("openfhe  imported OK:", hasattr(openfhe, "GenCryptoContext"))
PY' && ok "imports" || bad "imports"

echo "============================================================"
echo " 2/4  HE correctness (encrypted vs plaintext < 1e-3)"
echo "============================================================"
./run.sh 'python -m he.tests.test_pipeline --stage multi_iter --n-samples 3 --n-iters 1 --bootstrap --dims 37 138 1 --seed 42 --ring-dim-exp 15' \
    && ok "HE correctness" || bad "HE correctness"

echo "============================================================"
exp="0.6648"; [[ "${GPU:-0}" == "1" ]] && exp="0.6685"
echo " 3/4  plaintext parity: Pri-SAFL Cox-CC 2L  (canonical $exp)"
echo "============================================================"
out="$(./run.sh 'python ablation/main.py --loss cox-cc --n-layers 2 --approx-cox-cc 3-15cT4 --lr 0.045 --output verify_coxcc2' 2>&1)"
echo "$out" | tail -3
med="$(echo "$out" | grep -oE 'Median C-index: [0-9.]+' | grep -oE '[0-9.]+$' | tail -1)"
if [[ -n "$med" ]]; then
    awk -v m="$med" -v e="$exp" 'BEGIN{d=m-e; if(d<0)d=-d; printf ">> measured %.4f vs %.4f (|d|=%.4f, tol 0.005)\n", m, e, d; exit !(d<=0.005)}' \
        && ok "plaintext parity ($med)" || bad "plaintext drift ($med)"
else
    bad "plaintext parity (no median parsed)"
fi

echo "============================================================"
echo " 4/4  HE runtime smoke + timing"
echo "============================================================"
./run.sh 'python -m he.main --n-samples 3 --epochs 2 --bootstrap --timing --dims 37 138 1 --seed 42 --ring-dim-exp 15' \
    && ok "HE runtime smoke" || bad "HE runtime smoke"

echo "============================================================"
echo " summary: $pass passed, $fail failed"
echo "============================================================"
exit $((fail>0))
