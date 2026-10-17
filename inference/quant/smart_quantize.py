"""Port of the reference smart_quantize.py, re-ported verbatim: do not modify it
here -- changes go in the reference first. Notes plus interleaved silences form
one head-to-tail chain; overlapping notes make it infeasible (a failure).

Score = importance ladder M(p) (beat 0; 8th/16th/32nd/64th = -0.05/-0.1/-0.2/
-0.4*lambda; else -0.8*lambda) minus distance terms, maximized by the s0/s1
column DP and backtracked from the last row's best cell.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

# Any uniform scale works; fixed high-res integers keep the math exact.
QUARTER = 705_600_000             # one quarter note (the beat)
GRID = 22_050_000                 # QUARTER // 32, the default correction grid (a 128th note)
BAND = 16                         # default search band for durations / windows (±16 columns)
BAND_SPAN = BAND * GRID           # the default band expressed in time
NEG_INF = -1.0e10                 # DP initial value
FAIL_THRESHOLD = -1.0e9           # best score below this → failure
TICKS_PER_QUARTER = QUARTER


def lam_from_simplicity(s: float) -> float:
    """Strength slider s in [0,5] → weight lambda = 0.5 × 10^(s/5) (log scale)."""
    return 0.5 * (10.0 ** (s / 5.0))


def metric_bonus(abs_pos: int, lam: float) -> float:
    """Rhythmic-importance ladder M(p); p = absolute position mod QUARTER."""
    p = abs_pos % QUARTER
    if p == 0:
        return 0.0
    if p % (QUARTER // 2) == 0:
        return -0.05 * lam
    if p % (QUARTER // 4) == 0:
        return -0.1 * lam
    if p % (QUARTER // 8) == 0:
        return -0.2 * lam
    if p % (QUARTER // 16) == 0:
        return -0.4 * lam
    return -0.8 * lam


def metric_level(abs_pos: int) -> str:
    """Debug helper: name of the beat level a position sits on."""
    p = abs_pos % QUARTER
    if p == 0:
        return "beat"
    if p % (QUARTER // 2) == 0:
        return "8th"
    if p % (QUARTER // 4) == 0:
        return "16th"
    if p % (QUARTER // 8) == 0:
        return "32nd"
    if p % (QUARTER // 16) == 0:
        return "64th"
    return "128th"


@dataclass
class Result:
    ok: bool
    notes: List[Tuple[int, int]] = field(default_factory=list)   # new (start, end)
    detail: List[dict] = field(default_factory=list)             # per-record debug info
    message: str = ""


def smart_quantize(
    notes: Sequence[Tuple[int, int]],
    simplicity: float = 0.0,
    grid: int = GRID,
    band: Optional[int] = None,
) -> Result:
    """Core algorithm. notes: [(start, end), ...] in the same uniform tick
    scale as QUARTER; grid is the correction grid in that scale and band
    defaults to BAND_SPAN // grid (the same musical window).
    """
    lam = lam_from_simplicity(simplicity)
    if not notes:
        return Result(ok=False, message="empty selection")
    if grid <= 0:
        return Result(ok=False, message="invalid grid")
    if band is None:
        band = max(1, BAND_SPAN // grid)

    # 1. Record chain (sort + gap records)
    order = sorted(range(len(notes)), key=lambda i: notes[i][0])
    records: List[Tuple[int, int, Optional[int]]] = []
    for i in order:
        s, e = notes[i]
        e = max(s, e)
        if records and s - records[-1][1] >= grid / 2:
            records.append((records[-1][1], s, None))   # gap record
        records.append((s, e, i))
    R = len(records)

    # 2. Anchor + total column count
    first = records[0][0]
    if first >= 0:
        anchor = (first + grid // 2) // grid * grid
    else:
        anchor = -(((-first) + grid // 2) // grid * grid)
    rel = [(s - anchor, e - anchor) for s, e, _ in records]
    span = rel[-1][1]
    if span > 0:
        col_count = max(1, (span + grid // 2) // grid)      # round-half-up
    elif span < 0:
        col_count = -((-span + grid // 2) // grid)
    else:
        col_count = 0
    if col_count <= 0:
        return Result(ok=False, message="degenerate span")

    # Legal column windows (±band, clipped to [0, col_count))
    lo = [max(0, r0 // grid - band) for r0, _ in rel]
    hi = [min(col_count, r1 // grid + band) for _, r1 in rel]
    hi = [max(l, h) for l, h in zip(lo, hi)]

    S0: List[dict] = [dict() for _ in range(R)]     # best score with record r STARTING at column c
    S1: List[dict] = [dict() for _ in range(R)]     # best score with record r ENDING at column c
    BP: List[dict] = [dict() for _ in range(R)]     # s1 backtrack: start column of record r

    dur_ticks = [r1 - r0 for r0, r1 in rel]

    for r in range(R):
        wlo, whi = lo[r], hi[r]
        dcols = dur_ticks[r] // grid
        j_lo = max(1, dcols - band)
        j_hi = max(j_lo + 1, dcols + band)
        for c in range(wlo, whi):
            if r == 0:
                if c == 0:
                    d = abs(0 - rel[0][0])                      # anchor rounding residual
                    s0 = metric_bonus(anchor, lam) - 0.5 * (d / QUARTER) ** 2
                else:
                    s0 = NEG_INF
            else:
                prev = S1[r - 1].get(c - 1, NEG_INF)
                pos = anchor + grid * c
                d = abs(grid * c - rel[r][0])
                s0 = prev + metric_bonus(pos, lam) - 0.5 * (d / QUARTER) ** 2
            S0[r][c] = s0 if s0 > NEG_INF else NEG_INF

            # s1[r][c]: best start column within the j band
            best, best_start = NEG_INF, None
            for j in range(j_lo, j_hi):
                scol = c - j + 1
                if scol < wlo:
                    break                                       # start column shrinks as j grows
                cand = S0[r].get(scol, NEG_INF)
                if cand <= NEG_INF:
                    continue
                dd = abs(grid * j - dur_ticks[r])
                cand -= (dd / QUARTER) ** 2                         # duration distortion cost
                if cand > best:
                    best, best_start = cand, scol
            end_pos = anchor + grid * (c + 1)                   # exclusive end
            d_end = abs(grid * (c + 1) - rel[r][1])
            s1 = best - 0.5 * (d_end / QUARTER) ** 2 + metric_bonus(end_pos, lam)
            S1[r][c] = s1 if s1 > NEG_INF else NEG_INF
            BP[r][c] = best_start

    # 6. Free final end: backtrack from best cell of the last row
    last_col, final = max(S1[R - 1].items(), key=lambda kv: kv[1])
    if final < FAIL_THRESHOLD:
        return Result(
            ok=False,
            message=("quantize failed: overlapping notes make the "
                     "rhythmic chain infeasible"),
        )

    start_cols = [0] * R
    end_col = last_col
    for r in range(R - 1, -1, -1):
        sc = BP[r][end_col]
        if sc is None:
            return Result(ok=False, message="backtrack failed")
        start_cols[r] = sc
        end_col = sc - 1

    # 7. Write-back (head-to-tail)
    new_pairs = []
    for r in range(R):
        start = anchor + grid * start_cols[r]
        if r < R - 1:
            end = anchor + grid * start_cols[r + 1]
        else:
            end = anchor + grid * (last_col + 1)
        new_pairs.append((start, end))

    out: List[Tuple[int, int]] = [(0, 0)] * len(notes)
    detail = []
    for r, (_, _, idx) in enumerate(records):
        if idx is not None:
            out[idx] = new_pairs[r]
        detail.append({
            "record": r,
            "note_index": idx,                     # None = gap record
            "orig": records[r][:2],
            "new": new_pairs[r],
            "start_col": start_cols[r],
            "start_level": metric_level(new_pairs[r][0]),
        })

    moved = sum(1 for r, (_, _, idx) in enumerate(records)
                if idx is not None and new_pairs[r] != records[r][:2])
    return Result(
        ok=True,
        notes=out,
        detail=detail,
        message=f"lambda={lam_from_simplicity(simplicity):.4g}, {moved}/{len(notes)} notes moved",
    )


def smart_quantize_musical(
    notes: Sequence[Tuple[int, int]],
    simplicity: float = 0.0,
    ticks_per_quarter: int = 480,
    grid_ticks: Optional[int] = None,
) -> Result:
    """Any PPQ scale → internal scale → quantize → convert back; grid_ticks is
    the correction grid in the caller's tick scale (a 1/64 note at PPQ 480 is
    30) and None keeps the engine default (a 128th note).
    """
    scale = Fraction(TICKS_PER_QUARTER, ticks_per_quarter)
    grid = GRID if grid_ticks is None else max(1, int(round(grid_ticks * scale)))
    eng = [(int(Fraction(s) * scale), int(Fraction(e) * scale)) for s, e in notes]
    res = smart_quantize(eng, simplicity=simplicity, grid=grid)
    if res.ok:
        inv = Fraction(ticks_per_quarter, TICKS_PER_QUARTER)
        res.notes = [(int(Fraction(s) * inv), int(Fraction(e) * inv)) for s, e in res.notes]
        for d in res.detail:
            d["orig"] = tuple(int(Fraction(x) * inv) for x in d["orig"])
            d["new"] = tuple(int(Fraction(x) * inv) for x in d["new"])
    return res
