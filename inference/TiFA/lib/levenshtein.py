"""Sequence alignment via standard Levenshtein DP.

Cost is 0 for equality and 1 for substitution.  Backtrack prefers match over
delete over insert.
"""

from typing import TypeVar

T = TypeVar("T")


def levenshtein_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein edit distance between two sequences.

    Cost is 0 for equality and 1 for substitution/indel.
    """
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(prev + cost, dp[j] + 1, dp[j - 1] + 1)
            prev = tmp
    return dp[m]


def align_sequences(
        orig: list[T],
        mut: list[T],
) -> tuple[list[T | None], list[T | None]]:
    """Align *orig* to *mut* via Levenshtein DP.

    Returns ``(aligned_orig, aligned_mut)``, both same length, with ``None``
    marking gaps.
    """
    n, m = len(orig), len(mut)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if orig[i - 1] == mut[j - 1] else 1
            match_val = dp[i - 1][j - 1] + cost
            delete_val = dp[i - 1][j] + 1
            insert_val = dp[i][j - 1] + 1

            if match_val <= delete_val and match_val <= insert_val:
                dp[i][j] = match_val
            elif delete_val <= insert_val:
                dp[i][j] = delete_val
            else:
                dp[i][j] = insert_val

    al_orig: list[T | None] = []
    al_mut: list[T | None] = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if orig[i - 1] == mut[j - 1] else 1
            match_ok = dp[i][j] == dp[i - 1][j - 1] + cost
        else:
            match_ok = False
        if match_ok:
            al_orig.append(orig[i - 1])
            al_mut.append(mut[j - 1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            al_orig.append(orig[i - 1])
            al_mut.append(None)
            i -= 1
        else:
            al_orig.append(None)
            al_mut.append(mut[j - 1])
            j -= 1

    al_orig.reverse()
    al_mut.reverse()
    return al_orig, al_mut


def _build_consensus(rows: list[list[T | None]]) -> list[T | None]:
    """Column-wise consensus from a row-oriented profile (all rows same length)."""
    n_cols = len(rows[0])
    consensus: list[T | None] = []
    for col_idx in range(n_cols):
        counts: dict[T, int] = {}
        for row in rows:
            t = row[col_idx]
            if t is not None:
                counts[t] = counts.get(t, 0) + 1
        consensus.append(max(counts, key=counts.get) if counts else None)
    return consensus


def _merge_profile(
        rows: list[list[T | None]], new_path: list[T]
) -> list[list[T | None]]:
    """Align *new_path* to the existing profile and merge it in."""
    consensus = _build_consensus(rows)
    al_c, al_p = align_sequences(consensus, new_path)

    # Update existing rows: insert None where consensus had a gap
    new_rows: list[list[T | None]] = []
    for old_row in rows:
        new_row: list[T | None] = []
        old_idx = 0
        for c_tok in al_c:
            if c_tok is not None:
                new_row.append(old_row[old_idx])
                old_idx += 1
            else:
                new_row.append(None)
        new_rows.append(new_row)
    new_rows.append(al_p)
    return new_rows


def align_multiple_sequences(paths: list[list[T]]) -> list[list[T | None]]:
    """Align paths while preserving every source row, including duplicates."""
    if not paths:
        return []
    rows = [list(paths[0])]
    for path in paths[1:]:
        rows = _merge_profile(rows, path)
    return rows
