#!/usr/bin/env python3
"""GTA VI trophy rarity forecaster.

Model: a cohort mix. Every tracked player belongs to the cohort that joined
when they first synced; a cohort's share holding a trophy grows toward a
ceiling c like 1 - exp(-age / tau), with one (c, tau) pair fitted per trophy
from the case's own history. The tracked population P(t) is rebuilt hourly
from the reads and projected to the target with a log-log power law, so a
growing population dilutes rates (new cohorts hold few trophies) while old
cohorts keep progressing. The median is the cohort mix at the target; the
spread is a logit standard deviation combining a random-walk term, a
population-growth uncertainty term, the per-trophy fit residual, and a
refresh-risk term for horizons where the history shows no growth yet.

Stdlib only. Deterministic: no randomness, no network, no clock reads
except the case's own timestamps.
"""

import json
import math
import os
import sys

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
# Earn-curve timescales tried per trophy, in hours.
TAUS = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0)
# Prior timescale by tier, used only when the history cannot tell taus apart.
TIER_TAU = {"platinum": 504.0, "gold": 240.0, "silver": 120.0, "bronze": 60.0}


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(p):
    p = clamp(p, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    l = clamp(l, -9.210340371976184, 9.210340371976184)
    return 100.0 / (1.0 + math.exp(-l))


def parse_time(s):
    # ISO 8601 UTC, e.g. 2026-11-19T03:00:00Z (fractional seconds allowed).
    return _parse(s)


def _parse(s):
    from datetime import datetime, timezone
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).timestamp()


def fit_power_law(xs, ys):
    """Least squares y = a + b*x; returns (a, b). Needs >= 2 points."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    horizon_min = float(case["horizon_minutes"])
    target_ts = parse_time(case["target"])
    n = len(trophies)

    # ---- time base: hours since one hour before the first read ----
    read_ts = [parse_time(r["at"]) for r in history]
    origin = min(read_ts) - 3600.0 if read_ts else target_ts
    t_last = max(read_ts) if read_ts else target_ts
    horizon_h = (target_ts - t_last) / 3600.0
    total_h = int(math.ceil((target_ts - origin) / 3600.0)) + 1

    # ---- hourly tracked population P[h] ----
    p_obs = [float(r["players_tracked"]) for r in history]
    h_obs = [(t - origin) / 3600.0 for t in read_ts]
    p_hour = []
    for h in range(total_h):
        t = origin + h * 3600.0
        if not read_ts:
            p_hour.append(1.0)
        elif t <= read_ts[0]:
            p_hour.append(max(1.0, p_obs[0]))
        elif t >= read_ts[-1]:
            p_hour.append(max(1.0, p_obs[-1]))
        else:
            # log-linear interpolation between bracketing reads
            j = 0
            while j + 1 < len(read_ts) and read_ts[j + 1] < t:
                j += 1
            t0, t1 = read_ts[j], read_ts[j + 1]
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            lp = math.log(max(1.0, p_obs[j])) * (1 - f) + \
                math.log(max(1.0, p_obs[j + 1])) * f
            p_hour.append(max(1.0, math.exp(lp)))
    # Project P beyond the last read with a log-log power law on distinct reads.
    px, py = [], []
    seen_p = set()
    for h, p in zip(h_obs, p_obs):
        if p not in seen_p:
            seen_p.add(p)
            px.append(math.log1p(max(0.0, h)))
            py.append(math.log(max(1.0, p)))
    if len(px) >= 2:
        a, b = fit_power_law(px, py)
        b = clamp(b, 0.0, 2.0)
        last_h = (t_last - origin) / 3600.0
        for h in range(total_h):
            if origin + h * 3600.0 > t_last:
                x = math.log1p(max(0.0, (origin + h * 3600.0 - origin) / 3600.0))
                proj = math.exp(a + b * x)
                p_hour[h] = clamp(proj, p_hour[h], p_obs[-1] * 100.0)
    p_last = max(1.0, p_obs[-1]) if p_obs else 1.0
    p_target = p_hour[-1]
    growth = math.log(max(1.0, p_target / p_last))

    # ---- cohort joins per hour (newly tracked players) ----
    joins = [0.0] * total_h
    prev = 0.0
    for h in range(total_h):
        joins[h] = max(0.0, p_hour[h] - prev)
        prev = p_hour[h]

    # ---- cohort weight W[tau][hour]: earn rate with ceiling c = 1 ----
    # Incremental update: E(h+1) = E(h) + (1 - exp(-1/tau)) * S(h),
    # S(h+1) = exp(-1/tau) * (S(h) + joins[h+1]).
    weights = {}
    for tau in TAUS:
        dec = math.exp(-1.0 / tau)
        earn = 0.0
        s = 0.0
        w = [0.0] * total_h
        for h in range(total_h):
            if h > 0:
                earn += (1.0 - dec) * s
                s = dec * s + joins[h]
            else:
                s = joins[0]
            w[h] = earn / p_hour[h] if p_hour[h] > 0 else 0.0
        weights[tau] = w

    def hour_index(ts):
        return int(round((ts - origin) / 3600.0))

    read_h = [clamp(hour_index(t), 0, total_h - 1) for t in read_ts]
    target_h = clamp(hour_index(target_ts), 0, total_h - 1)

    # ---- per-trophy fit and forecast ----
    d_days = horizon_min / 1440.0
    out = []
    for ti, tr in enumerate(trophies):
        tier = tr.get("tier", "bronze")
        # Observed rates as fractions, skipping nulls.
        obs = []
        for ri, r in enumerate(history):
            v = r["rates"][ti] if ti < len(r["rates"]) else None
            if v is not None:
                obs.append((read_h[ri], clamp(float(v), 0.0, 100.0) / 100.0))
        prior_tau = TIER_TAU.get(tier, 60.0)
        if not obs:
            med = 0.005
            sd = 1.5
        else:
            best = None  # (sse, log-distance to prior, tau, c)
            for tau in TAUS:
                w = weights[tau]
                sw2 = sum(w[h] * w[h] for h, _ in obs)
                if sw2 <= 1e-12:
                    continue
                c = sum(w[h] * y for h, y in obs) / sw2
                c = clamp(c, 0.0001, 1.0)
                sse = sum((c * w[h] - y) ** 2 for h, y in obs)
                dist = abs(math.log(tau / prior_tau))
                key = (sse, dist)
                if best is None or key < best[0]:
                    best = (key, tau, c)
            if best is None:
                med = obs[-1][1]
                sd = 1.0
            else:
                _, tau, c = best
                w = weights[tau]
                med = clamp(c * w[target_h], 0.0001, 0.9999)
                # A refresh the history has not seen yet dilutes rates: new
                # cohorts hold few trophies, so the expected rate is pulled
                # toward the fresh-player level. The pull fades where the
                # population projection already models the dilution, and
                # vanishes for saturated trophies (fresh players hold them
                # too). Applied as a logit shift scaled by headroom.
                p_refresh = 1.0 - math.exp(-max(0.0, horizon_h) / 5.0)
                shift = -0.6 * p_refresh * math.exp(-growth) * (1.0 - med)
                med = clamp(1.0 / (1.0 + math.exp(
                    -(math.log(med / (1.0 - med)) + shift))), 0.0001, 0.9999)
                # Logit-space residual of the fit.
                res = []
                for h, y in obs:
                    pred = clamp(c * w[h], 0.0001, 0.9999)
                    yy = clamp(y, 0.0001, 0.9999)
                    res.append(math.log(pred / (1 - pred)) -
                               math.log(yy / (1 - yy)))
                resid = math.sqrt(sum(x * x for x in res) / len(res)) if res else 0.5
                resid = clamp(resid, 0.15, 1.0)
                # Spread: random walk + population-growth uncertainty +
                # fit residual + refresh risk where growth is unobserved.
                refresh = 0.7 * (1.0 - math.exp(-max(0.0, horizon_h) / 5.0)) \
                    * math.exp(-growth)
                var = 0.15 * d_days + (0.25 * growth) ** 2 + resid ** 2 + refresh ** 2
                sd = clamp(math.sqrt(var), 0.25, 2.0)
        med_pct = med * 100.0
        lm = logit_pct(med_pct)
        qs = []
        for z in Z:
            q = inv_logit_pct(lm + z * sd)
            qs.append(round(clamp(q, 0.0, 100.0), 4))
        # Enforce non-decreasing after rounding.
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        out.append({"id": int(tr["id"]), "q": qs})
    out.sort(key=lambda r: r["id"])
    return {"trophies": out}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    if not names:
        return 0
    for name in names:
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        result = forecast_case(case)
        stem = name[:-3]
        with open(os.path.join(out_dir, stem + ".out"), "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
