import math
import time

import torch
import torch.nn as nn
import transformers
from utils import mixed_quantizer, quantizer, quantizer_moe
from texttable import Texttable
from utils.reconstruct import torch_snr_error

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class GPTQ:
    # Backend selector for fasterquant. "pytorch" = original MC-MoE path.
    # "triton" / "triton_graph" = ported hbmpq Triton kernel (see
    # kernel/gptq_triton.py). 1-bit always uses pytorch because the binary
    # quantizer in utils/quantizer_moe.py is not yet ported.
    triton_backend: str = "pytorch"
    # Inner rank-1 propagation policy for the Triton path:
    #   None  = legacy MC-MoE semantics (apply only when wbits > 3)
    #   True  = always apply (standard GPTQ)
    #   False = never apply
    force_inner_rank1 = None
    # Quantize-formula form for Triton path. The default matches MC-MoE's
    # `clamp(round(x/s + z), 0, maxq)` and is required when zero is fractional
    # (which `_find_group_params_inplace` produces — see kernel/gptq_triton.py).
    # Flip to False (`round(x/s) + z` form) only with integer zero.
    force_add_z_before_round: bool = True
    # Debug knobs for numerics audits (no algorithmic effect at fp32 precision):
    force_fp64_div: bool = True
    force_outer_matmul: bool = False

    def __init__(self, layer, logger, name, wbits):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.quantizer = quantizer_moe.Quantizer()
        self.logger = logger
        self.name = name
        self.wbits = wbits
    
    def set_bit(self, bit):
        self.wbits = bit

    def add_batch(self, inp, out):
        # Hessian H = 2 X XT + λ I
        self.inp1 = None
        self.out1 = None

        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(self.layer.kernel_size, dilation=self.layer.dilation, padding=self.layer.padding, stride=self.layer.stride)
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def print_loss(self, name, q_weight, weight_error, timecost, modules=None, bit=3):
        table = Texttable()
        name = name+"-"+str(bit)
        name += ' ' * (31 - len(name))

        table.header(['name', 'weight_error', 'fp_inp_SNR', 'q_inp_SNR', 'time'])

        # assign weight
        if modules is None:
            self.layer.weight.data = q_weight.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        else:
            modules.weight.data = q_weight.reshape(modules.weight.shape).to(modules.weight.data.dtype)

        if self.inp1 is not None:
            # quantize input to int8
            quantizer = quantizer.Quantizer()
            quantizer.configure(8, perchannel=False, sym=True, mse=False)
            quantizer.find_params(self.inp1)
            q_in = quantizer.quantize(self.inp1).type(torch.float16)
            q_out = self.layer(q_in)

            # get kinds of SNR
            q_SNR = torch_snr_error(q_out, self.out1).item()
            fp_SNR = torch_snr_error(self.layer(self.inp1), self.out1).item()
        else:
            q_SNR = '-'
            fp_SNR = '-'
        table.set_cols_width([31, 10, 10, 10, 7])
        table.add_row([name, weight_error, fp_SNR, q_SNR, timecost])
        print(table.draw().split('\n')[-2])

    def static_fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, name=''):

        self.layer.to(self.dev)

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        W_orig_for_loss = W.clone()    # snapshot for true GPTQ loss (post-quant)

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)
        H = self.H
        H_orig_for_loss = H.clone()    # snapshot before Cholesky chain mutates H
        # Dump (W_fp16, H, wbits, name) for target sublayers when env var set.
        # MCMOE_DUMP_TARGET may be comma-separated list of substrings to match.
        import os
        _dump_targets = [t.strip() for t in os.environ.get('MCMOE_DUMP_TARGET', '').split(',') if t.strip()]
        if _dump_targets and any(t in name for t in _dump_targets):
            _dump_dir = os.environ.get('MCMOE_DUMP_DIR', './mcmoe_dump')
            os.makedirs(_dump_dir, exist_ok=True)
            _fname = f"{_dump_dir}/{name.replace('.', '_')}.pt"
            torch.save({
                'W_fp16': self.layer.weight.data.detach().cpu(),
                'H_fp32': H_orig_for_loss.detach().cpu(),
                'name': name,
                'wbits': self.wbits,
                'sym': bool(getattr(self.quantizer, 'sym', False)),
                'rows': self.rows, 'columns': self.columns,
            }, _fname)
            print(f'[dump] {name} → {_fname} (W shape={tuple(self.layer.weight.data.shape)}, H shape={tuple(H_orig_for_loss.shape)})')
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        W_orig_for_loss[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        g_idx = []
        scale = []
        zero = []
        now_idx = 1

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                    if ((i1 + i) // groupsize) - now_idx == -1:
                        scale.append(self.quantizer.scale)
                        zero.append(self.quantizer.zero)
                        now_idx += 1
                if self.quantizer.pack:
                    q, s, z = self.quantizer.quantize(w.unsqueeze(1))
                    q_r = s * (q - z)
                    q_r = q_r.flatten()
                    q = q.flatten()
                    Q1[:, i] = q
                else:
                    q_r = self.quantizer.quantize(w.unsqueeze(1))
                    q_r = q_r.flatten()
                    Q1[:, i] = q_r

                Losses1[:, i] = (w - q_r)**2 / d**2
                err1 = (w - q_r) / d

                if self.wbits > 3:
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        error = torch.sum(Losses).item()

        groupsize = groupsize if groupsize != -1 else self.columns
        g_idx = [i // groupsize for i in range(self.columns)]
        g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]
            g_idx = g_idx[invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        # True GPTQ objective on the un-mutated Hessian.
        delta = (W_orig_for_loss - (Q.t() if isinstance(self.layer, transformers.Conv1D) else Q)).float()
        true_L = float(((delta @ H_orig_for_loss) * delta).sum().item())
        print(f"[true_L] {name} wbits={self.wbits} L={true_L:.6e}")
        self.print_loss(name=name, q_weight=Q, weight_error=error, timecost=(time.time() - tick), bit=self.wbits)
        if scale == []:
            scale.append(self.quantizer.scale)
            zero.append(self.quantizer.zero)
    
        scale = torch.cat(scale, dim=1)
        zero = torch.cat(zero, dim=1)

        # z1 = zero.repeat_interleave(128, dim=1)
        # s1 = scale.repeat_interleave(128, dim=1)
        # print(Q.shape, scale.shape, zero.shape)
        # print(Q, zero, scale, zero.dtype)
        # self.dequant = s1 * (Q - z1)
        # print("W from gptq dequant", self.dequant)
        return scale, zero, g_idx, error

    def fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, name=''):
        backend = type(self).triton_backend
        if backend == 'pytorch':
            return self.static_fasterquant(blocksize, percdamp, groupsize, actorder, name)
        use_graph = backend == 'triton_graph'
        return self.triton_fasterquant(
            blocksize=blocksize, percdamp=percdamp, groupsize=groupsize,
            actorder=actorder, name=name, use_graph=use_graph,
            apply_inner_rank1=type(self).force_inner_rank1,
        )

    def triton_fasterquant(self, blocksize=128, percdamp=.01, groupsize=-1,
                           actorder=False, name='', use_graph=False,
                           apply_inner_rank1=None):
        """Triton dispatch. Same return signature as static_fasterquant
        (scale, zero, g_idx, error).

        apply_inner_rank1:
          None  → legacy MC-MoE behavior (skip when wbits <= 3)
          True  → always apply  (standard GPTQ)
          False → never apply

        The Triton path uses fp32 scale/qzero (vs MC-MoE pytorch path's fp16),
        and the reported error is weight-space MSE rather than GPTQ's
        Hessian-weighted Losses sum. Numerics will differ by ~1 ULP in
        scale-derived quant decisions; for ablation comparisons these
        should be negligible.
        """
        from kernel.gptq_triton import (
            fasterquant_inner_loop_triton,
            fasterquant_inner_loop_triton_graph,
        )

        if actorder:
            raise NotImplementedError("Triton backend does not support act_order")

        self.layer.to(self.dev)

        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W_orig = W.float().contiguous()
        W = W_orig.clone()

        tick = time.time()

        H = self.H
        H_orig_for_loss = H.clone()    # snapshot before Cholesky chain mutates H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0
        W_orig[:, dead] = 0
        H_orig_for_loss[dead, dead] = 1   # keep PD on dead columns; W_orig is already zeroed there

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H.contiguous()

        if groupsize == -1:
            groupsize = self.columns
        assert self.columns % groupsize == 0, (
            f"columns ({self.columns}) not divisible by groupsize ({groupsize})"
        )
        assert groupsize == blocksize, (
            "Triton path uses dynamic groups; requires groupsize == blocksize "
            f"(got {groupsize} vs {blocksize})"
        )
        n_groups = self.columns // groupsize
        scale = torch.zeros(self.rows, n_groups, dtype=W.dtype, device=W.device)
        qzero = torch.zeros(self.rows, n_groups, dtype=W.dtype, device=W.device)

        if apply_inner_rank1 is None:
            apply_inner_rank1 = (self.wbits > 3)

        maxq_f = float(2 ** self.wbits - 1)
        sym = bool(getattr(self.quantizer, 'sym', False))
        is_binary = (self.wbits == 1)

        kernel_fn = (
            fasterquant_inner_loop_triton_graph if use_graph
            else fasterquant_inner_loop_triton
        )
        kernel_fn(
            W, Hinv, scale, qzero, maxq_f, groupsize, blocksize,
            dynamic_groups=True, sym=sym,
            apply_inner_rank1=apply_inner_rank1,
            binary=is_binary,
            add_z_before_round=type(self).force_add_z_before_round,
            fp64_div=type(self).force_fp64_div,
            outer_use_matmul=type(self).force_outer_matmul,
        )
        torch.cuda.synchronize()

        error = float(((W_orig - W) ** 2).sum().item())

        g_idx = torch.tensor(
            [i // groupsize for i in range(self.columns)],
            dtype=torch.int32, device=W.device,
        )

        # True GPTQ objective on the un-mutated Hessian (uses dequant W, regardless
        # of pack mode — print_loss may swap W to int codes below).
        delta = (W_orig - W).float()
        true_L = float(((delta @ H_orig_for_loss) * delta).sum().item())
        print(f"[true_L] {name} wbits={self.wbits} L={true_L:.6e}")

        # Pack mode: recover integer codes from dequant W and per-group (scale,
        # qzero) so QLinear's bit-pack downstream sees integer values, matching
        # MC-MoE's pack=True branch (Q1[:, i] = q in static_fasterquant). The
        # recovery is exact in fp32 since W = s*(q - z) by construction.
        pack_mode = bool(getattr(self.quantizer, 'pack', False)) and not is_binary
        if pack_mode:
            n_groups = self.columns // groupsize
            W_view = W.view(self.rows, n_groups, groupsize)
            s_view = scale.unsqueeze(2)
            z_view = qzero.unsqueeze(2)
            int_q = torch.clamp(
                torch.round(W_view / s_view + z_view), 0.0, maxq_f
            ).view(self.rows, self.columns)
            Q = int_q
        else:
            Q = W

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.print_loss(
            name=name, q_weight=Q, weight_error=error,
            timecost=(time.time() - tick), bit=self.wbits,
        )
        # Return scale/zero in fp16 to match MC-MoE pytorch convention and
        # match downstream QLinear / HQQ kernel expectations.
        return scale.to(torch.float16), qzero.to(torch.float16), g_idx, error

    def free(self):
        self.inp1 = None
        self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()