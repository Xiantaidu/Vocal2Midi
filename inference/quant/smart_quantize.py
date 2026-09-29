"""Smart rhythmic alignment quantizer.

The algorithm mirrors the reference smart_quantize.py implementation and
must not be modified here: any change is made in the reference first and
re-ported verbatim. Only the reference's MIDI I/O and CLI are omitted.

Aligns the note boundaries of a monophonic melody (transcribed vocals,
recorded MIDI, etc.) to "rhythmically important" positions (on the beat,
8th, 16th, 32nd, 64th, ...) while changing the original performance as
little as possible. Rests between notes take part in the alignment as part
of the chain, so after quantization the whole phrase is contiguous and
every boundary lands on the correction grid.

The time scale is fully arbitrary: the algorithm only relies on ratios.
A quarter note (the beat) is QUARTER ticks and the correction grid is
GRID = QUARTER/32 (a 128th note); all distances are normalized by the
quarter length. The default constants use a high-resolution integer
scale (QUARTER = 705,600,000) purely to keep integer arithmetic exact;
``smart_quantize_musical`` converts from the caller's PPQ automatically.
There is no bar/measure awareness: the importance ladder is relative to
the beat.

simplicity (0~5) is the rhythmic simplification strength:
0 = conservative (barely move anything), 5 = aggressive (snap to strong
beats even if the move is large). lambda = 0.5 * 10^(s/5).

Algorithm outline
-----------------
1. Record chain: notes sorted in time; each silence between neighboring
   notes becomes a "gap record"; notes and gaps together form one
   head-to-tail chain. Overlapping notes are NOT clipped -- they make the
   chain infeasible and the quantization is reported as failed.
2. Anchor: the start of the first record is rounded to the nearest grid
   column (column = GRID step) and serves as the origin of the chain;
   the last column of the chain is forced to round(total span / GRID).
3. The only degree of freedom of each record is its "occupied column
   count j":
       max(1, dcols - 16) <= j <= dcols + 16,  dcols = floor(orig duration / GRID)
   The start column of record r = end column of record r-1 + 1.
4. Score (maximized):
       start:  M(pos_start)               - 0.5 * ((pos_start - orig_start)/QUARTER)^2
       dur:    - ((j*GRID - orig_dur)/QUARTER)^2
       end:    M(pos_end)                 - 0.5 * ((pos_end - orig_end)/QUARTER)^2
   M(p) is the rhythmic-importance ladder (p = absolute position mod QUARTER):
       on the beat            0
       8th note              -0.05*lambda
       16th note             -0.1 *lambda
       32nd note             -0.2 *lambda
       64th note             -0.4 *lambda
       anything else (grid)  -0.8 *lambda
5. Dynamic programming: two tables, s0[r][c] (record r STARTS at column c)
   and s1[r][c] (record r ENDS at column c); s0 has a single predecessor
   (s1[r-1][c-1], the chain is head-to-tail); s1 maximizes over the j
   band. Backtracking starts from the best cell of the last row (the
   final boundary is chosen freely by the DP, not forced); a best score
   below -1e9 is reported as failure (typical cause: heavily overlapping
   notes make the chain infeasible).
6. Write-back: every record boundary lands on a grid column and the end
   of each record equals the start of the next one.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Time constants (any uniform scale works; fixed integers keep math exact)
# --------------------------------------------------------------------------
QUARTER = 705_600_000             # one quarter note (the beat)
GRID = 22_050_000                 # QUARTER // 32, the correction grid (a 128th note)
BAND = 16                         # search band for durations / windows (±16 columns)
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
) -> Result:
    """
    Core algorithm. notes: [(start, end), ...] in any uniform tick scale,
    given in the same scale as the QUARTER/GRID constants (the MIDI path
    converts automatically).
    """
    lam = lam_from_simplicity(simplicity)
    if not notes:
        return Result(ok=False, message="empty selection")

    # ---- 1. Record chain (sort + gap records) ----
    order = sorted(range(len(notes)), key=lambda i: notes[i][0])
    records: List[Tuple[int, int, Optional[int]]] = []
    for i in order:
        s, e = notes[i]
        e = max(s, e)                       # clamp end to >= start
        if records and s > records[-1][1]:
            records.append((records[-1][1], s, None))   # gap record
        records.append((s, e, i))
    R = len(records)

    # ---- 2. Anchor and total column count ----
    first = records[0][0]
    if first >= 0:
        anchor = (first + GRID // 2) // GRID * GRID
    else:
        anchor = -(((-first) + GRID // 2) // GRID * GRID)
    rel = [(s - anchor, e - anchor) for s, e, _ in records]
    span = rel[-1][1]
    if span > 0:
        col_count = (span + GRID // 2) // GRID              # round-half-up
    elif span < 0:
        col_count = -((-span + GRID // 2) // GRID)
    else:
        col_count = 0
    if col_count <= 0:
        return Result(ok=False, message="degenerate span")

    # ---- Legal column windows (±16-column band, clipped to [0, colCount)) ----
    lo = [max(0, r0 // GRID - BAND) for r0, _ in rel]           # python // = floor
    hi = [min(col_count, r1 // GRID + BAND) for _, r1 in rel]
    hi = [max(l, h) for l, h in zip(lo, hi)]

    # ---- 3/4/5. DP ----
    S0: List[dict] = [dict() for _ in range(R)]     # best score with record r STARTING at column c
    S1: List[dict] = [dict() for _ in range(R)]     # best score with record r ENDING at column c
    BP: List[dict] = [dict() for _ in range(R)]     # s1 backtrack: start column of record r

    dur_ticks = [r1 - r0 for r0, r1 in rel]

    for r in range(R):
        wlo, whi = lo[r], hi[r]
        dcols = dur_ticks[r] // GRID
        j_lo = max(1, dcols - BAND)
        j_hi = dcols + BAND                                     # exclusive
        for c in range(wlo, whi):
            # ---- s0[r][c] ----
            if r == 0:
                if c == 0:
                    d = abs(0 - rel[0][0])                      # anchor rounding residual
                    s0 = metric_bonus(anchor, lam) - 0.5 * (d / QUARTER) ** 2
                else:
                    s0 = NEG_INF
            else:
                prev = S1[r - 1].get(c - 1, NEG_INF)
                pos = anchor + GRID * c
                d = abs(GRID * c - rel[r][0])
                s0 = prev + metric_bonus(pos, lam) - 0.5 * (d / QUARTER) ** 2
            S0[r][c] = s0 if s0 > NEG_INF else NEG_INF

            # ---- s1[r][c]: pick the start column within the j band ----
            best, best_start = NEG_INF, None
            for j in range(j_lo, j_hi):
                scol = c - j + 1
                if scol < wlo:
                    break                                       # start column shrinks as j grows
                cand = S0[r].get(scol, NEG_INF)
                if cand <= NEG_INF:
                    continue
                dd = abs(GRID * j - dur_ticks[r])
                cand -= (dd / QUARTER) ** 2                         # duration distortion cost
                if cand > best:
                    best, best_start = cand, scol
            end_pos = anchor + GRID * (c + 1)                   # exclusive end
            d_end = abs(GRID * (c + 1) - rel[r][1])
            s1 = best - 0.5 * (d_end / QUARTER) ** 2 + metric_bonus(end_pos, lam)
            S1[r][c] = s1 if s1 > NEG_INF else NEG_INF
            BP[r][c] = best_start

    # ---- 6. Free final end: backtrack from the best cell of the last row ----
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
        end_col = sc - 1                                       # end column of previous record

    # ---- 7. Write-back (head-to-tail; last end = chosen by the DP) ----
    new_pairs = []
    for r in range(R):
        start = anchor + GRID * start_cols[r]
        if r < R - 1:
            end = anchor + GRID * start_cols[r + 1]
        else:
            end = anchor + GRID * (last_col + 1)
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


# --------------------------------------------------------------------------
# Convenience wrapper: call directly in a common DAW tick scale (e.g. 480/quarter)
# --------------------------------------------------------------------------
def smart_quantize_musical(
    notes: Sequence[Tuple[int, int]],
    simplicity: float = 0.0,
    ticks_per_quarter: int = 480,
) -> Result:
    """
    Any PPQ scale → internal scale → quantize → convert back.
    Quantized boundaries sit on the 32nd grid, so as long as PPQ is a
    multiple of 8 the round-trip is exact.
    """
    scale = Fraction(TICKS_PER_QUARTER, ticks_per_quarter)
    eng = [(int(Fraction(s) * scale), int(Fraction(e) * scale)) for s, e in notes]
    res = smart_quantize(eng, simplicity=simplicity)
    if res.ok:
        inv = Fraction(ticks_per_quarter, TICKS_PER_QUARTER)
        res.notes = [(int(Fraction(s) * inv), int(Fraction(e) * inv)) for s, e in res.notes]
        for d in res.detail:
            d["orig"] = tuple(int(Fraction(x) * inv) for x in d["orig"])
            d["new"] = tuple(int(Fraction(x) * inv) for x in d["new"])
    return res
