"""Arabic character metadata used by ArabGlyphDiag.

The default table follows the 28-class ordering used in the AHCD/Hijja2
experiments. The structural taxonomy is operational and hierarchical; it is
not a causal model of handwriting or recognition behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Mapping


@dataclass(frozen=True)
class GlyphMetadata:
    id: int
    name: str
    glyph: str
    skeleton_group: str
    dot_count: int
    dot_position: str

    def to_dict(self) -> dict:
        return asdict(self)


ARABIC_LETTERS: List[GlyphMetadata] = [
    GlyphMetadata(0, "alef", "ا", "alef", 0, "none"),
    GlyphMetadata(1, "beh", "ب", "beh", 1, "below"),
    GlyphMetadata(2, "teh", "ت", "beh", 2, "above"),
    GlyphMetadata(3, "theh", "ث", "beh", 3, "above"),
    GlyphMetadata(4, "jeem", "ج", "jeem", 1, "below"),
    GlyphMetadata(5, "hah", "ح", "jeem", 0, "none"),
    GlyphMetadata(6, "khah", "خ", "jeem", 1, "above"),
    GlyphMetadata(7, "dal", "د", "dal", 0, "none"),
    GlyphMetadata(8, "thal", "ذ", "dal", 1, "above"),
    GlyphMetadata(9, "reh", "ر", "reh", 0, "none"),
    GlyphMetadata(10, "zain", "ز", "reh", 1, "above"),
    GlyphMetadata(11, "seen", "س", "seen", 0, "none"),
    GlyphMetadata(12, "sheen", "ش", "seen", 3, "above"),
    GlyphMetadata(13, "sad", "ص", "sad", 0, "none"),
    GlyphMetadata(14, "dad", "ض", "sad", 1, "above"),
    GlyphMetadata(15, "tah", "ط", "tah", 0, "none"),
    GlyphMetadata(16, "zah", "ظ", "tah", 1, "above"),
    GlyphMetadata(17, "ain", "ع", "ain", 0, "none"),
    GlyphMetadata(18, "ghain", "غ", "ain", 1, "above"),
    GlyphMetadata(19, "feh", "ف", "feh", 1, "above"),
    GlyphMetadata(20, "qaf", "ق", "feh", 2, "above"),
    GlyphMetadata(21, "kaf", "ك", "kaf", 0, "none"),
    GlyphMetadata(22, "lam", "ل", "lam", 0, "none"),
    GlyphMetadata(23, "meem", "م", "meem", 0, "none"),
    GlyphMetadata(24, "noon", "ن", "noon", 1, "above"),
    GlyphMetadata(25, "heh", "ه", "heh", 0, "none"),
    GlyphMetadata(26, "waw", "و", "waw", 0, "none"),
    GlyphMetadata(27, "yeh", "ي", "yeh", 2, "below"),
]

DEFAULT_METADATA: Dict[int, GlyphMetadata] = {m.id: m for m in ARABIC_LETTERS}
DOTTED_CLASS_IDS = tuple(m.id for m in ARABIC_LETTERS if m.dot_position != "none")


def normalize_metadata(metadata: Mapping[int, object] | Iterable[object] | None = None) -> Dict[int, GlyphMetadata]:
    """Normalize custom metadata into an ``id -> GlyphMetadata`` mapping."""
    if metadata is None:
        return dict(DEFAULT_METADATA)

    if isinstance(metadata, Mapping):
        items = list(metadata.values())
    else:
        items = list(metadata)

    out: Dict[int, GlyphMetadata] = {}
    for item in items:
        if isinstance(item, GlyphMetadata):
            m = item
        elif isinstance(item, Mapping):
            m = GlyphMetadata(
                id=int(item["id"]),
                name=str(item["name"]),
                glyph=str(item.get("glyph", "")),
                skeleton_group=str(item["skeleton_group"]),
                dot_count=int(item["dot_count"]),
                dot_position=str(item["dot_position"]),
            )
        else:
            raise TypeError("Metadata items must be GlyphMetadata or mapping objects.")
        if m.dot_position not in {"none", "above", "below"}:
            raise ValueError(f"Unsupported dot_position for class {m.id}: {m.dot_position}")
        if m.id in out:
            raise ValueError(f"Duplicate class id in metadata: {m.id}")
        out[m.id] = m

    if not out:
        raise ValueError("Metadata cannot be empty.")
    return out
