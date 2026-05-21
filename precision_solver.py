import pickle
import os
import argparse

import torch
import gurobipy as gp


# Effective storage bit per nominal bit (g=128). 1-bit/sym: only fp16 scale
# overhead (16/128 = 0.125). 2-bit/asym and 3-bit/asym: fp16 scale + fp16
# zero overhead (2 * 16/128 = 0.25). These match the result.md `eff_bit_moe`
# column accounting.
DEFAULT_EFF_BITS = {1: 1.125, 2: 2.25, 3: 3.25}


def _parse_eff_bits(s: str) -> dict:
    out = {}
    for tok in s.split(","):
        k, v = tok.split(":")
        out[int(k)] = float(v)
    return out


def _parse_target_avgs(s: str) -> list:
    return [float(t) for t in s.split(",") if t.strip()]


class experts_ilp():
    def __init__(self,
                 actnum_path,
                 x_space=(1, 2, 3),
                 num_experts=8,
                 quant_loss_path=None,
                 weight_path=None,
                 alpha=1,
                 beta=1,
                 gama=1,
                 norm_experts=False,
                 ):
        self.x_space = x_space
        self.num_experts = num_experts

        with open(actnum_path, 'rb') as file:
            actnum_matrix = pickle.load(file)
        with open(quant_loss_path, 'rb') as file:
            quant_loss_matrix = pickle.load(file)
        with open(weight_path, 'rb') as file:
            weight_matrix = pickle.load(file)

        self.blocks = list(actnum_matrix.keys())
        scale_factor = 1
        if norm_experts:
            actnum_matrix = self.norm_experts_dim(actnum_matrix)
            weight_matrix = self.norm_experts_dim(weight_matrix)
            scale_factor = 1000
        self.loss_matrix = {}
        for i in self.blocks:
            i_loss_matrix = {}
            for j in range(self.num_experts):
                j_loss_matrix = {}
                expert_significance = actnum_matrix[i][j] ** alpha * weight_matrix[i][j] ** beta
                for x in self.x_space:
                    j_loss_matrix[x] = expert_significance * quant_loss_matrix[i][j][x] ** alpha * scale_factor
                i_loss_matrix[j] = j_loss_matrix
            self.loss_matrix[i] = i_loss_matrix

    # ---- old per-block ILP (kept for backward compatibility) ----

    def bulid_ilp_model(self, nblock, constrait):
        loss_matrix = self.loss_matrix[nblock]
        lp_content = "Minimize\nOBJ"
        lp_content += "\nSubject To\n"
        lp_content += " + ".join(f"y{i}" for i in range(1, self.num_experts + 1)) + " - OBJ = 0\n"
        lp_content += " + ".join(f"1 x{i}_{1} + 2 x{i}_{2} + 3 x{i}_{3}" for i in range(1, self.num_experts + 1)) + f" <= {constrait}\n"
        lp_content += " + ".join(f"x{i}_{3}" for i in range(1, self.num_experts + 1)) + f" >= 1\n"
        lp_content += " + ".join(f"x{i}_{2}" for i in range(1, self.num_experts + 1)) + f" >= 1\n"
        for i in range(1, self.num_experts + 1):
            lp_content += f"y{i} - " + " - ".join(f"{loss_matrix[i-1][j]} x{i}_{j}" for j in self.x_space) + " = 0\n"
            lp_content += f" + ".join(f"x{i}_{j}" for j in self.x_space) + " = 1\n"
        lp_content += "Binary\n"
        lp_content += " ".join(f"x{i}_{j}" for i in range(1, self.num_experts + 1) for j in self.x_space)
        return lp_content

    def solve_ilp_model(self, model_path):
        model = gp.read(model_path)
        model.optimize()
        opt_set = []
        for v in model.getVars():
            if v.VarName.startswith('x'):
                if v.X == 1:
                    opt_set.append(int(v.VarName[-1]))
        experts_keys = list(range(self.num_experts))
        opt_set_dict = dict(zip(experts_keys, opt_set))
        return opt_set_dict

    def expert2tensor(self, expert_dict):
        experts_tensor = torch.tensor(list(expert_dict.values()))
        return experts_tensor

    def norm_experts_dim(self, x):
        norm_x = {}
        for i in self.blocks:
            if not torch.is_tensor(x[i]):
                experts_tensor = self.expert2tensor(x)
            else:
                experts_tensor = x[i]
            norm_experts = experts_tensor / float(experts_tensor.sum())
            norm_x[i] = norm_experts
        return norm_x

    def ilp_solver(self, constrait):
        final_opt_set = {}
        for n in self.blocks:
            lp_model = self.bulid_ilp_model(n, constrait)
            with open('model.lp', 'w') as file:
                file.write(lp_model)
            opt_set = self.solve_ilp_model('model.lp')
            final_opt_set[n] = opt_set
        return final_opt_set

    # ---- new global ILP with effective storage bits ----

    def global_eff_ilp_solver(self, target_eff_avg: float, eff_bits: dict | None = None):
        """Global ILP across all (block, expert) pairs.

        Minimizes total loss s.t. the layer-wise average effective storage bit
        equals ``target_eff_avg``. Effective bit per nominal choice k is given
        by ``eff_bits[k]`` (defaults to {1: 1.125, 2: 2.25, 3: 3.25} — fp16
        scale (+zero for asym) overhead at group size 128). No per-block bit-2
        / bit-3 diversity constraints are imposed so that low targets such as
        T=1.5 remain feasible.

        Returns a {block_idx: {expert_idx: nominal_bit}} dict, matching the
        per-block solver output format (consumable by main.py).
        """
        if eff_bits is None:
            eff_bits = dict(DEFAULT_EFF_BITS)

        n_blocks = len(self.blocks)
        n_experts = self.num_experts
        n_total = n_blocks * n_experts
        budget = target_eff_avg * n_total

        model = gp.Model("global_eff_ilp")
        model.Params.OutputFlag = 0

        # x[block, expert, k] ∈ {0,1}
        x = {}
        for nb in self.blocks:
            for e in range(n_experts):
                for k in self.x_space:
                    x[(nb, e, k)] = model.addVar(vtype=gp.GRB.BINARY,
                                                 name=f"x_{nb}_{e}_{k}")

        # Objective: minimize total weighted loss.
        model.setObjective(
            gp.quicksum(self.loss_matrix[nb][e][k] * x[(nb, e, k)]
                        for nb in self.blocks
                        for e in range(n_experts)
                        for k in self.x_space),
            sense=gp.GRB.MINIMIZE,
        )

        # Exactly one bit choice per (block, expert).
        for nb in self.blocks:
            for e in range(n_experts):
                model.addConstr(
                    gp.quicksum(x[(nb, e, k)] for k in self.x_space) == 1,
                    name=f"one_{nb}_{e}",
                )

        # Global effective-bit budget.
        #
        #   sum eff_bits[k] * x[block, expert, k]  ≤  T * n_total
        #
        # Why ≤ and not ==: the objective is `sum loss[…,k] * x[…,k]` with
        # loss monotonically decreasing in k (higher bit → lower quant_loss
        # → lower weighted loss). So the optimum naturally saturates the
        # budget to within one transition step (smallest step = 3→2 = 1.0 in
        # the ×8 scale, i.e. 0.125 in eff-bit units). Forcing == would
        # exclude lower-loss patterns that happen to leave < 1 transition of
        # slack and produce a *worse* solution at the same realized avg.
        #
        # The post-hoc assertion below ensures the realized avg is in fact
        # within one transition step of T — if not, either the loss matrix
        # is degenerate or the formulation is wrong.
        budget_eff = target_eff_avg * n_total
        model.addConstr(
            gp.quicksum(eff_bits[k] * x[(nb, e, k)]
                        for nb in self.blocks
                        for e in range(n_experts)
                        for k in self.x_space) <= budget_eff,
            name="eff_bit_budget",
        )

        model.optimize()
        if model.Status != gp.GRB.OPTIMAL:
            raise RuntimeError(
                f"Gurobi did not return OPTIMAL (status={model.Status}) for "
                f"target_eff_avg={target_eff_avg}"
            )

        sol = {}
        realized = 0.0
        for nb in self.blocks:
            sol[nb] = {}
            for e in range(n_experts):
                picked = None
                for k in self.x_space:
                    if x[(nb, e, k)].X > 0.5:
                        picked = k
                        break
                assert picked is not None
                sol[nb][e] = picked
                realized += eff_bits[picked]
        realized_avg = realized / n_total
        # Largest gap between consecutive eff-bit values bounds the maximum
        # slack a saturating <=-optimum can leave (for {1.125, 2.25, 3.25}:
        # max step = 2.125, min step = 1.0). Realized avg must lie in
        # [T - max_step / n_total, T]. Violation indicates a loss matrix
        # that is not monotonic in k — bail loudly rather than silently
        # produce a misnamed pkl.
        step_vals = sorted(eff_bits.values())
        max_step = step_vals[-1] - step_vals[0]
        lo = target_eff_avg - max_step / n_total - 1e-9
        if not (lo <= realized_avg <= target_eff_avg + 1e-9):
            raise RuntimeError(
                f"global_eff_ilp_solver: realized_avg={realized_avg:.6f} "
                f"outside [{lo:.6f}, {target_eff_avg:.6f}] — loss matrix "
                f"may not be monotonic in bit width."
            )
        return sol, realized_avg


def get_args_parser():
    parser = argparse.ArgumentParser('Set ilp configs', add_help=False)
    parser.add_argument('--actnum_path', default='experts_act_frequency.pkl', type=str)
    parser.add_argument('--quant_loss_path', default='experts_quant_loss.pkl', type=str)
    parser.add_argument('--weight_path', default='experts_act_weight.pkl', type=str)
    parser.add_argument('--save_path', default='experts_mixture_bit_selection', type=str)
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--beta', default=1.5, type=float)
    parser.add_argument('--gama', default=2, type=float)
    # Mode-specific args.
    parser.add_argument('--mode', default='block', choices=['block', 'global_eff'],
                        help="'block' = legacy per-block ILP with nominal {1,2,3} "
                             "bits and start/end_bitwidth budgets. 'global_eff' = "
                             "global ILP with effective bits (1.125/2.25/3.25), "
                             "targeting layer-wise average T.")
    # Legacy 'block' args.
    parser.add_argument('--start_bitwidth', default=12, type=int)
    parser.add_argument('--end_bitwidth', default=21, type=int)
    # 'global_eff' args.
    parser.add_argument('--target_avgs', type=str,
                        default='1.5,1.625,1.75,1.875,2.0,2.125,2.25,2.375,2.5,2.625,2.75',
                        help="Comma-separated layer-wise average effective bits to solve for.")
    parser.add_argument('--eff_bits', type=str,
                        default='1:1.125,2:2.25,3:3.25',
                        help="Comma-separated nominal:effective mapping (e.g. "
                             "'1:1.125,2:2.25,3:3.25' for sym/asym/asym g=128).")
    parser.add_argument('--num_experts', default=8, type=int)
    return parser


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Experts bit selection with ilp', parents=[get_args_parser()])
    args = parser.parse_args()
    experts_ilp_example = experts_ilp(args.actnum_path,
                                      quant_loss_path=args.quant_loss_path,
                                      weight_path=args.weight_path,
                                      num_experts=args.num_experts,
                                      alpha=args.alpha,
                                      beta=args.beta,
                                      gama=args.gama,
                                      norm_experts=True)
    os.makedirs(args.save_path, exist_ok=True)

    if args.mode == 'block':
        for i in range(args.start_bitwidth, args.end_bitwidth):
            opt_set = experts_ilp_example.ilp_solver(i)
            total_bits = str(i)
            save_name = f"experts_mixture_bitwidth_combination_{total_bits}bit.pkl"
            with open(os.path.join(args.save_path, save_name), 'wb') as f:
                pickle.dump(opt_set, f)
    elif args.mode == 'global_eff':
        eff_bits = _parse_eff_bits(args.eff_bits)
        targets = _parse_target_avgs(args.target_avgs)
        for T in targets:
            opt_set, realized = experts_ilp_example.global_eff_ilp_solver(T, eff_bits=eff_bits)
            save_name = f"experts_mixture_bitwidth_effavg_{T:.3f}.pkl"
            save_path = os.path.join(args.save_path, save_name)
            with open(save_path, 'wb') as f:
                pickle.dump(opt_set, f)
            print(f"[solver] target_eff_avg={T:.3f}  realized={realized:.6f}  "
                  f"→ {save_path}")
