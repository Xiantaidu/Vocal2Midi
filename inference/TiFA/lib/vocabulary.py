import json
import pathlib
from types import MappingProxyType
from typing import Iterable, Mapping, TypeVar

# PAD = 0 (implicitly defined)
MASK_TOKEN = 1
SPACE_TOKEN = 2

NUM_RESERVED_TOKENS = max(MASK_TOKEN, SPACE_TOKEN) + 1

__all__ = [
    "MASK_TOKEN",
    "SPACE_TOKEN",
    "NUM_RESERVED_TOKENS",
    "qualify_symbol",
    "is_stop_symbol",
    "VocabularyBuilder",
    "Vocabulary",
]


def qualify_symbol(
    symbol: str,
    default_language: str | None,
    global_symbols: Iterable[str],
) -> str:
    """Apply the vocabulary's language-prefix rule to one symbol."""
    if symbol not in global_symbols and "/" not in symbol and default_language is not None:
        return f"{default_language}/{symbol}"
    return symbol


def is_stop_symbol(
    symbol: str,
    default_language: str | None,
    global_symbols: Iterable[str],
    stop_symbols: Iterable[str],
) -> bool:
    """Return whether a raw or language-qualified symbol is a stop."""
    if symbol in stop_symbols:
        return True
    return (
        qualify_symbol(
            symbol,
            default_language,
            global_symbols,
        )
        in stop_symbols
    )


class VocabularyBuilder:
    def __init__(
            self,
            *,
            global_symbols: Iterable[str] = (),
            stop_symbols: Iterable[str] = (),
            merged_groups: Iterable[Iterable[str]] | None = None,
            prebuilt_vocab: "Vocabulary | None" = None,
    ):
        self.global_symbols = frozenset(global_symbols)
        self.stop_symbols = frozenset(stop_symbols)
        self.merged_groups = [list(g) for g in (merged_groups or ())]
        self.prebuilt_vocab = prebuilt_vocab
        self._symbol_counts: dict[str, int] = {}
        if prebuilt_vocab is not None:
            self._symbol_counts.update({s: 0 for s in prebuilt_vocab.symbol_to_id})

    def add(self, symbols: Iterable[str], default_language: str | None) -> None:
        for s in symbols:
            if is_stop_symbol(
                s,
                default_language,
                self.global_symbols,
                self.stop_symbols,
            ):
                continue
            s = qualify_symbol(s, default_language, self.global_symbols)
            self._symbol_counts[s] = self._symbol_counts.get(s, 0) + 1

    def counter(self) -> Mapping[str, int]:
        return MappingProxyType(self._symbol_counts)

    def build(self) -> "Vocabulary":
        if self.prebuilt_vocab is not None:
            return self._build_from_prebuilt()
        return self._build_from_scratch()

    def _build_from_scratch(self) -> "Vocabulary":
        observed = set(self._symbol_counts.keys())
        for i, members in enumerate(self.merged_groups):
            members = [str(s) for s in members]
            for s in members:
                if s in self.stop_symbols:
                    raise ValueError(
                        f"Stop symbol '{s}' cannot be a member of merged group {i}.")
            if not any(s in observed for s in members):
                raise ValueError(
                    f"None of the symbols in merged group {i} "
                    f"appear in the dataset: [{', '.join(members)}]")

        # Collect all non-stop symbols including merged group members
        all_symbols = observed.copy()
        for members in self.merged_groups:
            for s in members:
                if s not in self.stop_symbols:
                    all_symbols.add(str(s))

        groups = _disjoint_sets(
            all_symbols,
            ([str(s) for s in members if str(s) not in self.stop_symbols]
             for members in self.merged_groups),
        )
        groups.sort(key=lambda g: g[0])

        symbol_to_id: dict[str, int] = {}
        for idx, group in enumerate(groups):
            gid = idx + NUM_RESERVED_TOKENS
            for s in group:
                symbol_to_id[s] = gid

        return Vocabulary(symbol_to_id=symbol_to_id)

    def _build_from_prebuilt(self) -> "Vocabulary":
        symbol_to_id = self.prebuilt_vocab.symbol_to_id
        observed = set(self._symbol_counts.keys())

        # Validate merged groups: stop symbols are still forbidden
        for i, members in enumerate(self.merged_groups):
            members = [str(s) for s in members]
            for s in members:
                if s in self.stop_symbols:
                    raise ValueError(
                        f"Stop symbol '{s}' cannot be a member of merged group {i}.")

        # Collect all symbols involved: new observations plus all merged
        # group members (including pre-built ones, to connect components).
        new_symbols = {s for s in observed if s not in symbol_to_id}
        all_symbols = new_symbols.copy()
        for members in self.merged_groups:
            for s in members:
                s = str(s)
                if s not in self.stop_symbols:
                    all_symbols.add(s)

        if not all_symbols:
            return Vocabulary(symbol_to_id=symbol_to_id)

        groups = _disjoint_sets(
            all_symbols,
            ([str(s) for s in members if str(s) not in self.stop_symbols]
             for members in self.merged_groups),
        )
        groups.sort(key=lambda g: g[0])

        prebuilt_ids = list(symbol_to_id.values())
        next_id = max(prebuilt_ids) + 1 if prebuilt_ids else NUM_RESERVED_TOKENS

        for group in groups:
            prebuilt_in_group = [symbol_to_id[s] for s in group if s in symbol_to_id]
            if prebuilt_in_group:
                target = min(prebuilt_in_group)
            else:
                target = next_id
                next_id += 1
            for s in group:
                if s not in symbol_to_id:
                    symbol_to_id[s] = target

        return Vocabulary(symbol_to_id=symbol_to_id)


class Vocabulary:
    def __init__(
            self,
            *,
            symbol_to_id: dict[str, int],
    ):
        self._symbol_to_id = symbol_to_id
        id_to_symbols: dict[int, list[str]] = {}
        for sym, idx in symbol_to_id.items():
            id_to_symbols.setdefault(idx, []).append(sym)
        self._id_to_symbols: dict[int, tuple[str, ...]] = {
            idx: tuple(syms) for idx, syms in id_to_symbols.items()
        }

    @property
    def symbol_to_id(self) -> dict[str, int]:
        return dict(self._symbol_to_id)

    @property
    def vocab_size(self) -> int:
        ids = self._symbol_to_id.values()
        return max(NUM_RESERVED_TOKENS, *ids) + 1 if ids else NUM_RESERVED_TOKENS + 1

    def __len__(self) -> int:
        return self.vocab_size

    def encode(self, symbol: str, language: str | None) -> int | None:
        resolved = self.resolve(symbol, (language,) if language is not None else ())
        return resolved[0] if resolved is not None else None

    def resolve(self, symbol: str, languages: Iterable[str] = ()) -> tuple[int, str] | None:
        """Return the token ID and exact symbol, trying language prefixes in order."""
        if symbol in self._symbol_to_id:
            return self._symbol_to_id[symbol], symbol
        if "/" in symbol:
            return None
        for language in languages:
            prefixed_symbol = f"{language}/{symbol}"
            if prefixed_symbol in self._symbol_to_id:
                return self._symbol_to_id[prefixed_symbol], prefixed_symbol
        return None

    def decode(self, token: int, stringfy: bool = False) -> "tuple[str, ...] | str":
        """Return the symbol(s) mapped to *token*.

        Returns a tuple of all symbols that share this ID (more than one
        when the ID represents a merged group).  If *stringfy* is True,
        joins them with ``, `` and returns a single string.
        Returns ``None`` for unknown tokens.
        """
        syms = self._id_to_symbols.get(token)
        if syms is None:
            return None
        if stringfy:
            return ", ".join(syms)
        return syms

    def to_dict(self) -> dict:
        return {
            "symbols": dict(self._symbol_to_id),
        }

    def dump(self, path: str | pathlib.Path) -> None:
        with open(path, "w", encoding="utf8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> "Vocabulary":
        with open(path, "r", encoding="utf8") as f:
            data = json.load(f)
        return cls(symbol_to_id=data["symbols"])


_T = TypeVar("_T")


def _disjoint_sets(
        elements: Iterable[_T],
        unions: Iterable[Iterable[_T]],
) -> list[tuple[_T, ...]]:
    """Partition *elements* by unioning each group in *unions*.

    Returns a list of sorted tuples, one per equivalence class.
    """
    parent: dict[_T, _T] = {}

    def find(x: _T) -> _T:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: _T, y: _T) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for e in elements:
        parent[e] = e

    for members in unions:
        members = list(members)
        if not members:
            continue
        first = members[0]
        if first not in parent:
            parent[first] = first
        for m in members[1:]:
            if m not in parent:
                parent[m] = m
            union(first, m)

    root_to_members: dict[_T, list[_T]] = {}
    for e in parent:
        r = find(e)
        root_to_members.setdefault(r, []).append(e)

    return [tuple(sorted(members)) for members in root_to_members.values()]
