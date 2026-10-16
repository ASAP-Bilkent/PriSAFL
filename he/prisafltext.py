import numpy as np
from openfhe import Ciphertext
from he.encryption import INTMPBOOT_INPUT_LEVELS, multiparty_decrypt
from he.utils import (profile, timer_context, is_power_of_two,
                      next_power_of_two, distance_to_next_power_of_two)

class PriSAFLText():
    ckks_context = None
    VERBOSE = 0

    @classmethod
    def set_ckks_context(cls, context):
        cls.ckks_context = context

    @classmethod
    def _decrypt(cls, ciphertext):
        if cls.ckks_context.n_clients > 1:
            return multiparty_decrypt(cls.ckks_context.cc, cls.ckks_context.parties, ciphertext)
        return cls.ckks_context.cc.Decrypt(cls.ckks_context.keys.secretKey, ciphertext)

    @classmethod
    def set_verbosity(cls, verbose):
        cls.VERBOSE = verbose

    def get_level(self):
        if self._cipher_data is not None:
            return self._cipher_data.GetLevel()
        return None

    @profile
    def __init__(self, inp, mask=None):
        if PriSAFLText.ckks_context is None:
            raise ValueError("CKKS context not set. Call PriSAFLText.set_ckks_context() before creating instances.")

        self._data = None
        self._padded_data = None
        self._plaintext_data = None
        self._cipher_data = None

        self.validation_mask = None
        self._valid_len = None
        self.shape = None

        self.init_state = None

        if isinstance(inp, Ciphertext):
            assert mask is not None, "mask must be provided for Ciphertext input"
            self._cipher_data = inp
            self.validation_mask = mask
            self.init_state = "C"
        else:
            self._data = inp

            if np.isscalar(self._data):
                self._data = np.array([[self._data]]).reshape(1, 1)

            self.n_rows, self.n_cols = self._data.shape

            self.shape = (self.n_rows, self.n_cols)
            self.init_state = "D"

        if self.shape is not None:
            self.__class__.verbose_print(f"*** Creating new PriSAFLText with shape {self.shape} ***", 2)
        else:
            self.__class__.verbose_print(f"*** Creating new PriSAFLText with valid_len {self.valid_len} ***", 2)

    @property
    def data(self):
        if self._data is None:
            self.__class__.verbose_print("No data", 4)
        return self._data

    @property
    def padded_data(self):
        if self._padded_data is None:
            self.__class__.verbose_print("No padded data", 4)

            if self.init_state == "D":
                raise ValueError("_pad_data() must be called manually for PriSAFLText with init_state == D")
            elif self.init_state == "C":
                if self.plaintext_data is not None:
                    self._plaintext_to_padded()
                    self.__class__.verbose_print("Created padded data", 4)
        return self._padded_data

    @property
    def plaintext_data(self):
        if self._plaintext_data is None:
            self.__class__.verbose_print("No plaintext data", 4)

            if self.init_state == "D":
                if self.padded_data is not None:
                    self._padded_to_plaintext()
                    self.__class__.verbose_print("Created plaintext data", 4)
            elif self.init_state == "C":
                if self._cipher_data is not None:
                    self._ciphertext_to_plaintext(self._cipher_data)
                    self.__class__.verbose_print("Created plaintext data", 4)
        return self._plaintext_data

    @property
    def cipher_data(self):
        if self._cipher_data is None:
            self.__class__.verbose_print("No cipher data", 4)
            if self.init_state == "D":
                if self.plaintext_data is not None:
                    self._plaintext_to_ciphertext()
                    self.__class__.verbose_print("Created ciphertext data", 4)

        return self._cipher_data

    @property
    def valid_len(self):
        if self.validation_mask is None:
            self._valid_len = self.n_rows * self.n_cols
        else:
            self._valid_len = len(self.validation_mask.flatten())
        return self._valid_len

    @data.setter
    def data(self, value):
        self._data = value

    @padded_data.setter
    def padded_data(self, value):
        self._padded_data = value

    @plaintext_data.setter
    def plaintext_data(self, value):
        self._plaintext_data = value

    @cipher_data.setter
    def cipher_data(self, value):
        if self.data is not None:
            raise ValueError("Cannot set 'cipher_data' when 'data' is present. Initialize PriSAFLText with ciphertext first, then set 'data'.")
        else:
            self._cipher_data = value

    def _return_decrypted(self):
        result = PriSAFLText._decrypt(self.cipher_data)
        result.SetLength(self.valid_len)
        return result.GetRealPackedValue()

    def _ciphertext_to_plaintext(self, ciphertext):
        result = PriSAFLText._decrypt(ciphertext)
        result.SetLength(self.valid_len)
        self._plaintext_data = result

    def _plaintext_to_padded(self):
        self._padded_data = np.array(self.plaintext_data.GetRealPackedValue())

    def _pad_data(self, order="row", N=None, M=None, pad_step=None, pad_batch=None):
        if self._padded_data is not None:
            raise ValueError("_pad_data() called, but _padded_data already exists.")

        if order == "row":
            order = "C"
        elif order == "col":
            order = "F"
        else:
            raise ValueError("order should be one of 'row' or 'col'.")

        if N is not None and M is not None and pad_step is not None:
            raise ValueError("Either supply N & M or pad-step & pad_batch, not both.")

        if not hasattr(self, 'n_rows') or not hasattr(self, 'n_cols'):
            if self.shape is not None:
                self.n_rows, self.n_cols = self.shape
            elif self._data is not None:
                self.n_rows, self.n_cols = self._data.shape
                self.shape = (self.n_rows, self.n_cols)

        if N is None and M is None and pad_step is None:
            self.padded_data = self.data.flatten(order=order)
            self.validation_mask = np.ones_like(self.padded_data)
        elif N is not None and M is not None:
            if self.n_rows == 1 and self.n_cols == 1:
                self.padded_data = np.full((N, M), self.data).flatten()
                self.validation_mask = np.ones_like(self.padded_data)
            else:
                original_shape = self.shape

                def _pad_dimension(is_row_padding):
                    dim_size = self.n_rows if is_row_padding else self.n_cols
                    pad_needed = not is_power_of_two(dim_size)

                    if pad_needed:
                        dist = distance_to_next_power_of_two(dim_size)
                        if is_row_padding:
                            self.n_rows += dist
                            pad_width = ((0, dist), (0, 0))
                        else:
                            self.n_cols += dist
                            pad_width = ((0, 0), (0, dist))
                        self.padded_data = np.pad(self.data, pad_width=pad_width, mode='constant', constant_values=0)
                    else:
                        self.padded_data = self.data

                _pad_dimension(is_row_padding=(order != "C"))

                self.shape = (self.n_rows, self.n_cols)
                if M == 1:
                    tile_factor = M
                else:
                    tile_factor = N * M // self.n_cols
                self.padded_data = np.tile(self._padded_data, tile_factor).flatten(order=order)

                mask = np.zeros(self.shape, dtype=int)
                mask[:original_shape[0], :original_shape[1]] = 1
                self.validation_mask = np.tile(mask, tile_factor).flatten(order=order)
        elif pad_step is not None:
            original_shape = self.shape

            def _apply_padding(is_row_padding):
                dim_size = self.n_rows if is_row_padding else self.n_cols

                if pad_step == dim_size:
                    self.padded_data = self.data
                else:
                    if pad_step > dim_size:
                        dist = pad_step - dim_size
                        if is_row_padding:
                            self.n_rows = pad_step
                        else:
                            self.n_cols = pad_step
                    else:
                        dist = pad_step

                    if is_row_padding:
                        pad_width = ((0, dist), (0, 0))
                    else:
                        pad_width = ((0, 0), (0, dist))

                    self.padded_data = np.pad(self.data, pad_width=pad_width, mode='constant', constant_values=0)

            is_row_padding = (order != "C")
            _apply_padding(is_row_padding)

            self.shape = (self.n_rows, self.n_cols)

            if pad_batch is not None:
                if pad_batch == 1:
                    tile_factor = pad_batch
                elif pad_step > 1:
                    tile_factor = pad_batch
                else:
                    tile_factor = pad_step * pad_batch // max(self.n_rows, self.n_cols)

                if pad_batch > 0:
                    self.padded_data = self.padded_data.flatten(order=order)
                    self.padded_data = np.tile(self.padded_data, abs(tile_factor))
                else:
                    self.padded_data = np.tile(self.padded_data, (abs(tile_factor), 1))
                    self.padded_data = self.padded_data.flatten(order=order)
            else:
                self.padded_data = self.padded_data.flatten(order=order)

            mask = np.zeros(self.shape, dtype=int)
            mask[:original_shape[0], :original_shape[1]] = 1
            self.validation_mask = mask.flatten(order=order)

            if pad_batch is not None:
                self.validation_mask = np.tile(self.validation_mask, abs(tile_factor))

    def _padded_to_plaintext(self):
        self._plaintext_data = PriSAFLText.ckks_context.cc.MakeCKKSPackedPlaintext(self.padded_data)

    def _plaintext_to_ciphertext(self):
        self._cipher_data = PriSAFLText.ckks_context.cc.Encrypt(PriSAFLText.ckks_context.keys.publicKey, self.plaintext_data)

    def apply_mult_types(self, other, mult_types):
        def partial_mult_type(X, mult_type):
            if mult_type == "C":
                self.__class__.verbose_print("Asking for ciphertext", 4)
                return X.cipher_data
            elif mult_type == "P":
                self.__class__.verbose_print("Asking for plaintext", 4)
                return X.plaintext_data
            elif mult_type == "D":
                self.__class__.verbose_print("Asking for data", 4)
                return X.data
            else:
                raise ValueError("mult_type must be one of 'C', 'P' or 'D'.")

        return partial_mult_type(self, mult_types[0]), partial_mult_type(other, mult_types[1])

    def negate(self):
        text = PriSAFLText(self.ckks_context.cc.EvalNegate(self.cipher_data), mask=self.validation_mask)
        text.shape = self.shape
        return text

    def add(self, other, mult_types="CC"):
        self_data, other_data = self.apply_mult_types(other, mult_types)
        text = PriSAFLText(self.ckks_context.cc.EvalAdd(self_data, other_data), mask=self.validation_mask)
        text.shape = self.shape
        return text

    def subtract(self, other, mult_types="CC"):
        self_data, other_data = self.apply_mult_types(other, mult_types)
        text = PriSAFLText(self.ckks_context.cc.EvalSub(self_data, other_data), mask=self.validation_mask)
        text.shape = self.shape
        return text

    @profile
    def mult(self, other, mult_types="CC"):
        self_data, other_data = self.apply_mult_types(other, mult_types)
        if other.validation_mask is not None:
            new_validation_mask = self.validation_mask * other.validation_mask
        else:
            new_validation_mask = self.validation_mask

        text = PriSAFLText(self.ckks_context.cc.EvalMult(self_data, other_data), mask=new_validation_mask)
        text.shape = self.shape
        return text

    @staticmethod
    def sum_ciphertexts(texts:list):
        n = len(texts)
        acc = texts[0].cipher_data
        if n > 1:
            for i in range(1, n):
                acc = PriSAFLText.ckks_context.cc.EvalAdd(acc, texts[i].cipher_data)
        result = PriSAFLText(acc, mask=texts[0].validation_mask)
        return result

    def multiply(self, other, mult_types="CC", rotate=None):
        if other.isscalar() and not self.isscalar():
            self, other = other, self

        if not self.isscalar() and not other.isscalar():
            result = self.matrix_multiply(other, mult_types, rotate)
        else:
            result = self.scalar_multiply(other, mult_types)
        return result

    def matrix_multiply(self, other, mult_types, rotate):
        if self._padded_data is None and self._cipher_data is None:
            self._pad_data("row", *other.shape)
        if other._padded_data is None and other._cipher_data is None:
            other._pad_data("col", *self.shape[::-1])

        J, N, M = self.shape[0], other.shape[0], other.shape[1]

        result = self.mult(other, mult_types)

        if rotate is None:
            rotate = [1, N]

        result = result._rotate_ciphertext("left", rotate[0], rotate[1])
        result.shape = (J, M)

        return result

    def scalar_multiply(self, other, mult_types):
        if self._padded_data is None and other._padded_data is None:
            self._pad_data("row", *other.shape)
            other._pad_data("row", *self.shape)
        elif self._padded_data is None or other._padded_data is None:
            raise ValueError("Either none or both PriSAFLTexts should have _padded_data")

        A_main, B_main = self.apply_mult_types(other, mult_types)
        result_enc = self.ckks_context.cc.EvalMult(A_main, B_main)
        result = PriSAFLText(result_enc, mask=self.validation_mask)
        return result

    def apply_bootstrapping_if_needed(self):
        if self._cipher_data is None:
            return
        return self.apply_bootstrapping()

    def apply_bootstrapping(self):
        with timer_context('Bootstrap'):
            if self.ckks_context.n_clients > 1:
                cc = self.ckks_context.cc
                parties = self.ckks_context.parties
                n_clients = self.ckks_context.n_clients

                ct = self._cipher_data
                remaining = self.ckks_context.depth - (ct.GetLevel() + ct.GetNoiseScaleDeg() - 1)
                if remaining < INTMPBOOT_INPUT_LEVELS:
                    raise RuntimeError(f"IntMPBoot needs >= {INTMPBOOT_INPUT_LEVELS} levels left "
                                       f"on its input, got {remaining}")

                inCtxt = cc.IntMPBootAdjustScale(self._cipher_data)
                a = cc.IntMPBootRandomElementGen(parties[0].publicKey)

                c1 = inCtxt.Clone()
                c1.RemoveElement(0)

                sharePairVec = []
                for i in range(n_clients):
                    sharePairVec.append(cc.IntMPBootDecrypt(parties[i].secretKey, c1, a))

                aggregatedSharesPair = cc.IntMPBootAdd(sharePairVec)

                self._cipher_data = cc.IntMPBootEncrypt(
                    self.ckks_context.keys.publicKey,
                    aggregatedSharesPair,
                    a,
                    inCtxt
                )
            else:
                num_iterations = 1
                precision = 17
                bootstrapped_cipher_data = self.ckks_context.cc.EvalBootstrap(self._cipher_data, num_iterations, precision)
                self._cipher_data = bootstrapped_cipher_data

    def _rotate_ciphertext(self, direction, pad_step, N, clean=True):
        if direction == "left":
            multiplier = 1
        elif direction == "right":
            multiplier = -1
        else:
            raise ValueError("direction could be one of 'left', 'right'.")

        logN = int(np.ceil(np.log2(N)))
        pad_step = multiplier * pad_step

        if logN < 1:
            return self

        ciphertext_result = self.cipher_data
        new_validation_mask = self.validation_mask

        for i in range(logN):
            self.__class__.verbose_print(f"pad_step/logn: {pad_step}, {i+1}/{logN}", 1)
            ciphertext_rotate = self.ckks_context.cc.EvalRotate(ciphertext_result, pad_step)
            ciphertext_result = self.ckks_context.cc.EvalAdd(ciphertext_result, ciphertext_rotate)

            if direction == "right":
                def rotate_mask_right(valmask, pad_step):
                    pad_step = abs(pad_step)
                    ones_indices = np.where(valmask == 1)[0]
                    rotated_indices = set(ones_indices).union(set(ones_indices + pad_step))

                    max_index = max(rotated_indices)
                    if max_index >= len(valmask):
                        extended_length = next_power_of_two(max_index + 1)
                        extended_array = np.zeros(extended_length, dtype=int)
                    else:
                        extended_array = np.zeros(len(valmask), dtype=int)

                    for idx in rotated_indices:
                        if idx < len(extended_array):
                            extended_array[idx] = 1

                    return extended_array

                new_validation_mask = rotate_mask_right(new_validation_mask, pad_step)
            else:
                def left_rotate_mask(mask, pad_step):
                    result = []
                    for i in range(0, len(mask), pad_step * 2):
                        result.extend(mask[i:i + pad_step])
                        result.extend([0] * pad_step)
                    return np.array(result)

                new_validation_mask = left_rotate_mask(new_validation_mask, pad_step)
            pad_step *= 2

        result = PriSAFLText(ciphertext_result, mask=new_validation_mask)
        result.shape = self.shape

        if direction == "left" and clean:
            mask = PriSAFLText(result.validation_mask.reshape(1, -1))
            mask.padded_data = mask.data.flatten(order="C")
            mask.validation_mask = np.ones_like(mask.padded_data)

            result = result.mult(mask, "CP")

        return result

    def isscalar(self):
        return self.shape == (1, 1)

    def dump_info(self, name=""):
        if PriSAFLText.VERBOSE > 0:
            np.set_printoptions(suppress=True, precision=2)
            self.__class__.verbose_print(f"{name}_data:{self.data}", 1)

            np.set_printoptions(threshold=4096)

            if self.__class__.VERBOSE >= 3:
                print(f"{name}_padded_data:")
                self.print_colored_array(self.padded_data * self.validation_mask, 32)

                print(f"{name}_vmask:")
                self.print_colored_array(self.validation_mask, 32, suffix=f"{len(self.validation_mask)}")
            else:
                mask = self.validation_mask
                if len(self.padded_data) != len(mask):
                    mask = mask[:len(self.padded_data)]
                self.__class__.verbose_print(f"{name}_padded_data:{self.padded_data*mask}", 1)
                self.__class__.verbose_print(f"{name}_vmask:{self.validation_mask}{len(self.validation_mask)}", 1)

            self.__class__.verbose_print(f"{name}_shape:{self.shape}", 1)
            if self._cipher_data is not None:
                self.__class__.verbose_print(f"{name}_level:{self._cipher_data.GetLevel()}", 1)
            np.set_printoptions(suppress=False)

    @staticmethod
    def print_colored_array(arr, n, prefix="", suffix=""):
        green = "\033[92m"
        red = "\033[91m"
        reset = "\033[0m"
        print(prefix, end="")
        for i in range(0, len(arr), n):
            color = green if (i // n) % 2 == 0 else red
            print(color + str(arr[i:i+n]) + reset, end=" ")
        print(suffix)

    @classmethod
    def verbose_print(cls, message, level):
        if cls.VERBOSE > 0 and cls.VERBOSE >= level:
            print(message)
        elif cls.VERBOSE < 0 and abs(cls.VERBOSE) == level:
            print(message)
