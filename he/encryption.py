import numpy as np
from openfhe import (CCParamsCKKSRNS, COMPRESSION_LEVEL, FHECKKSRNS, GenCryptoContext,
                     PKESchemeFeature, ScalingTechnique, SecretKeyDist, SecurityLevel,
                     get_native_int)

LEVELS_AFTER_BOOTSTRAP = 14
# IntMPBoot input needs 1 + SLACK(3) levels. With fewer, OpenFHE silently returns a corrupted ciphertext
INTMPBOOT_INPUT_LEVELS = 4
MAX_CONTEXT_TRIES = 16

def multiparty_decrypt(cc, parties, ciphertext):
    shares = [cc.MultipartyDecryptLead([ciphertext], parties[0].secretKey)[0]]
    shares += [cc.MultipartyDecryptMain([ciphertext], p.secretKey)[0] for p in parties[1:]]
    return cc.MultipartyDecryptFusion(shares)

def context_keyswitch_ok(cc, keys, decrypt=None):
    decrypt = decrypt or (lambda ct: cc.Decrypt(keys.secretKey, ct))
    try:
        n_slots = cc.GetRingDimension() // 2
        pt = cc.MakeCKKSPackedPlaintext([0.3] * n_slots)
        ct = cc.EvalMult(cc.Encrypt(keys.publicKey, pt), cc.Encrypt(keys.publicKey, pt))
        step = 1
        while step < n_slots:
            ct = cc.EvalAdd(ct, cc.EvalRotate(ct, step))
            step *= 2
        dec = decrypt(ct)
        dec.SetLength(n_slots)
        v = np.array(dec.GetRealPackedValue())
        return bool(np.isfinite(v).all()) and float(np.max(np.abs(v))) < 1e6
    except Exception:
        return False

class CKKSContext:
    def __init__(self, n_clients, ring_dim_exp=None):
        is_multiparty = n_clients > 1
        self.n_clients = n_clients
        self.cc, self.depth = self._setup_self(is_multiparty,
                                               ring_dim_exp=ring_dim_exp)

        for attempt in range(MAX_CONTEXT_TRIES):
            if is_multiparty:
                self._multiparty_keygen()
                decrypt = lambda ct: multiparty_decrypt(self.cc, self.parties, ct)
            else:
                self.keys = self.cc.KeyGen()
                self.cc.EvalMultKeyGen(self.keys.secretKey)
                ring_bits = self.cc.GetRingDimension().bit_length() - 1
                rot_steps = [1 << i for i in range(ring_bits)]
                self.cc.EvalRotateKeyGen(self.keys.secretKey, rot_steps)
                decrypt = None
            if context_keyswitch_ok(self.cc, self.keys, decrypt):
                break
            self.cc, self.depth = self._setup_self(is_multiparty,
                                                   ring_dim_exp=ring_dim_exp)
        else:
            raise RuntimeError(
                f"CKKS context key-switching invalid after {MAX_CONTEXT_TRIES} rebuilds")
        if not is_multiparty:
            self.cc.EvalBootstrapKeyGen(self.keys.secretKey, self.cc.GetRingDimension() // 2)

    def _multiparty_keygen(self):
        self.parties = []
        kp_prev = None
        for i in range(self.n_clients):
            kp = self.cc.KeyGen() if i == 0 else self.cc.MultipartyKeyGen(kp_prev.publicKey)
            if not kp.good():
                raise RuntimeError(f"Key generation failed for party {i}")
            self.parties.append(kp)
            kp_prev = kp
        self.keys = self.parties[-1]
        self._setup_multiparty_eval_keys()

    def _setup_multiparty_eval_keys(self):
        n = self.n_clients

        evalMultKeys = []
        evalMultKeys.append(self.cc.KeySwitchGen(self.parties[0].secretKey, self.parties[0].secretKey))

        for i in range(1, n):
            evalMultKeys.append(self.cc.MultiKeySwitchGen(
                self.parties[i].secretKey,
                self.parties[i].secretKey,
                evalMultKeys[0]
            ))

        evalMultAccum = evalMultKeys[0]
        for i in range(1, n):
            evalMultAccum = self.cc.MultiAddEvalKeys(
                evalMultAccum,
                evalMultKeys[i],
                self.keys.publicKey.GetKeyTag()
            )

        evalMultTransformed = []
        for i in range(n):
            evalMultTransformed.append(self.cc.MultiMultEvalKey(
                self.parties[i].secretKey,
                evalMultAccum,
                self.keys.publicKey.GetKeyTag()
            ))

        evalMultFinal = evalMultTransformed[0]
        for i in range(1, n):
            evalMultFinal = self.cc.MultiAddEvalMultKeys(
                evalMultFinal,
                evalMultTransformed[i],
                self.keys.publicKey.GetKeyTag()
            )

        self.cc.InsertEvalMultKey([evalMultFinal])

        ring_dim_exp = self.cc.GetRingDimension().bit_length() - 1
        rot_steps = [1 << i for i in range(ring_dim_exp)]
        rot_all = rot_steps + [-s for s in rot_steps]
        self.cc.EvalSumKeyGen(self.parties[0].secretKey)
        evalSumKeys = self.cc.GetEvalSumKeyMap(self.parties[0].secretKey.GetKeyTag())

        for i in range(1, n):
            evalSumKeysParty = self.cc.MultiEvalSumKeyGen(
                self.parties[i].secretKey,
                evalSumKeys,
                self.keys.publicKey.GetKeyTag()
            )
            evalSumKeys = self.cc.MultiAddEvalSumKeys(
                evalSumKeys,
                evalSumKeysParty,
                self.keys.publicKey.GetKeyTag()
            )

        self.cc.InsertEvalSumKey(evalSumKeys)

        self.cc.EvalRotateKeyGen(self.parties[0].secretKey, rot_all)
        evalRotateKeys = self.cc.GetEvalAutomorphismKeyMap(self.parties[0].secretKey.GetKeyTag())

        for i in range(1, n):
            evalRotateKeysParty = self.cc.MultiEvalAtIndexKeyGen(
                self.parties[i].secretKey,
                evalRotateKeys,
                rot_all,
                self.keys.publicKey.GetKeyTag()
            )
            evalRotateKeys = self.cc.MultiAddEvalAutomorphismKeys(
                evalRotateKeys,
                evalRotateKeysParty,
                self.keys.publicKey.GetKeyTag()
            )

        self.cc.InsertEvalAutomorphismKey(evalRotateKeys, "")

    def _setup_self(self, is_multiparty=False, ring_dim_exp=None):
        secret_key_dist = SecretKeyDist.UNIFORM_TERNARY

        parameters = CCParamsCKKSRNS()
        parameters.SetSecretKeyDist(secret_key_dist)

        parameters.SetSecurityLevel(SecurityLevel.HEStd_NotSet if ring_dim_exp
                                    else SecurityLevel.HEStd_128_classic)
        parameters.SetRingDim(1 << (ring_dim_exp or (16 if is_multiparty else 17)))
        if get_native_int() == 128:
            parameters.SetScalingModSize(78)
            parameters.SetScalingTechnique(ScalingTechnique.FIXEDAUTO)
            parameters.SetFirstModSize(89)
        else:
            parameters.SetScalingModSize(59)
            parameters.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
            parameters.SetFirstModSize(60)

        level_budget = [3, 3]
        if is_multiparty:
            depth = LEVELS_AFTER_BOOTSTRAP - 1 + INTMPBOOT_INPUT_LEVELS
        else:
            depth = LEVELS_AFTER_BOOTSTRAP + FHECKKSRNS.GetBootstrapDepth(level_budget, secret_key_dist)

        parameters.SetMultiplicativeDepth(depth)

        if is_multiparty:
            parameters.SetInteractiveBootCompressionLevel(COMPRESSION_LEVEL.SLACK)

        cc = GenCryptoContext(parameters)
        cc.Enable(PKESchemeFeature.PKE)
        cc.Enable(PKESchemeFeature.KEYSWITCH)
        cc.Enable(PKESchemeFeature.LEVELEDSHE)
        cc.Enable(PKESchemeFeature.ADVANCEDSHE)
        if is_multiparty:
            cc.Enable(PKESchemeFeature.MULTIPARTY)
        else:
            cc.Enable(PKESchemeFeature.FHE)
            cc.EvalBootstrapSetup(level_budget)

        return cc, depth
