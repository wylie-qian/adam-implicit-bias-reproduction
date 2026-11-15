"""Post-hoc diagnostics, 2026-11-12. NOT part of the frozen 2026-11-05 protocol.

Reuses run.py's rescaled Adam step on the retained 2026-11-05 datasets.
Usage: python3 -I diag.py OUT_DIR JOBS_JSON N_WORKERS
Each job: [name, seed, method, schedule, epsilon, epsilon_mode, steps]
  schedule ["poly1", a, c]: eta_t = c * t^(-a), t = 1, 2, ... (paper figure legends)
  schedule ["const", null, eta]: fixed eta (GD / GDM)
  epsilon_mode "true": epsilon at its real size (as run.py)
  epsilon_mode "rescaled": hypothesis only; epsilon added to the underflow-rescaled
  moments without rescaling epsilon itself (see README).
"""
import importlib.util, json, math, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy.special import expit, logsumexp

REPO = Path(__file__).resolve().parents[2]  # experiments/paper50
spec = importlib.util.spec_from_file_location("p50run", REPO / "run.py")
run = importlib.util.module_from_spec(spec); spec.loader.exec_module(run)
DATA = REPO / "results/2026-11-05/datasets"
OUT = Path(sys.argv[1])
B1, B2 = 0.9, 0.99

def ckpts(T):
    s = set(np.unique(np.round(np.logspace(0, math.log10(T), 400)).astype(int)).tolist())
    s.add(T); return s

def log_loss(m):
    # log of mean softplus(-m), stable for large m
    l = np.where(m > 30, -m, np.log(np.logaddexp(0.0, -m)))
    return float(logsumexp(l) - math.log(len(m)))

def one(job):
    name, seed, method, sched, eps, mode, T = job
    with np.load(DATA / f"seed-{seed:02d}" / "data.npz") as d:
        z = d["y"][:, None] * d["X"]; w = d["w0"].copy()
    ref = json.load(open(DATA / f"seed-{seed:02d}" / "reference.json"))["reference"]
    gi, g2 = ref["gamma_linf"], ref["gamma_l2"]
    m = np.zeros_like(w); v = np.zeros_like(w); scale = 0.0
    cp = ckpts(T); rec = []; t0 = time.time(); err = None
    kind, a, c = sched  # ('poly', a, c, offset) or ('const', eta)
    for t in range(T):
        if kind == "poly1":   eta = c * (t + 1) ** (-a)     # paper legend t^-a, t = 1, 2, ...
        elif kind == "poly2": eta = c * (t + 2) ** (-a)     # our frozen protocol
        else:                 eta = c
        try:
            if method == "adam":
                if mode == "true":
                    w, m, v, scale = run.scaled_adam_step(z, w, m, v, scale, B1, B2, eta, eps)
                else:  # hypothesis: eps added to the RESCALED moments (eps not scaled by e^scale)
                    w, m, v, scale = _resc(z, w, m, v, scale, eta, eps)
            else:
                g = -z.T @ expit(-(z @ w)) / len(z)
                if method == "gdm":   m = B1 * m + (1 - B1) * g; g = m
                elif method == "gdm_torch": m = B1 * m + g; g = m
                w = w - eta * g
        except Exception as e:
            err = repr(e); break
        k = t + 1
        if k in cp:
            mg = z @ w; mn = float(mg.min())
            qi = mn / np.abs(w).max(); q2 = mn / np.linalg.norm(w)
            row = dict(step=k, gap_linf=gi - qi, rel_linf=(gi - qi) / gi, gap_l2=g2 - q2, rel_l2=(g2 - q2) / g2,
                       log_loss=log_loss(mg), norm_linf=float(np.abs(w).max()))
            if method == "adam":
                raw_min_sqrt_v = float(np.log10(np.sqrt(v.min())) - scale / math.log(10)) if v.min() > 0 else None
                row["log10_min_sqrt_v_raw"] = raw_min_sqrt_v
            rec.append(row)
    json.dump(dict(name=name, seed=seed, method=method, sched=sched, eps=eps, mode=mode, T=T,
                   error=err, seconds=time.time() - t0, rows=rec), open(OUT / f"{name}.json", "w"))
    return name, err, time.time() - t0

def _resc(z, w, m, v, old_scale, eta, eps):
    margins = z @ w
    scale = max(0.0, float(np.min(margins)))
    h = np.exp(scale - np.logaddexp(0.0, margins))
    g = -z.T @ h / len(z)
    f = 1.0 if not np.any(v) and not np.any(m) else math.exp(scale - old_scale)
    m = B1 * f * m + (1 - B1) * g
    v = B2 * f * f * v + (1 - B2) * g * g
    return w - eta * m / (np.sqrt(v) + eps), m, v, scale

if __name__ == "__main__":
    jobs = [tuple(j) for j in json.load(open(sys.argv[2]))]
    with Pool(int(sys.argv[3])) as p:
        for r in p.imap_unordered(one, jobs):
            print(*r, flush=True)
