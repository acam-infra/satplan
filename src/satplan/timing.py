"""Power-frequency scaling and single-step decode iteration timing."""
from __future__ import annotations


def kv_bytes_per_tok(model: dict) -> int:
    return 2 * int(model["layers"]) * int(model["kv_heads"]) * int(model["head_dim"]) * int(model["bytes_per_elem"])


def clock_factor(p_cap: float, power: dict) -> float:
    dyn = max(p_cap - float(power["P_idle"]), 1e-9) / (float(power["P_max"]) - float(power["P_idle"]))
    f = dyn ** (1.0 / float(power["gamma"]))
    return min(max(f, float(power["f_min"])), 1.0)


def iter_time(nd: int, ctx: float, f: float, gpu: dict) -> float:
    beta = float(gpu["mem_freq_exp"])
    mem = float(gpu["t_weights_s"]) + float(gpu["t_kv_tok_s"]) * ctx
    return float(gpu["t_fixed_s"]) + mem / (f**beta) + (float(gpu["t_seq_s"]) * nd) / f
