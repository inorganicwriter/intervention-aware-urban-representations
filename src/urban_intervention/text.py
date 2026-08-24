"""Domain-neutral text normalization shared across package layers."""

from __future__ import annotations

import re


def normalize_station_name(value: object) -> str:
    """Return the frozen cross-source station-name matching key."""
    text = str(value).strip() if value is not None else ""
    if not text:
        return ""
    text = re.sub(r"站?[（(][^）)]*[）)]", "", text)
    text = text.translate(str.maketrans("", "", "（）()"))
    text = text.replace("·", "").replace("-", "")
    text = text.removesuffix("站").removesuffix("路")
    return re.sub(r"\s+", "", text).lower()
