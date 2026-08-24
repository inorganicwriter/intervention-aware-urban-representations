"""Station-name compatibility normalization."""

from urban_intervention.text import normalize_station_name as _normalize_station_name


def norm_station_name(name) -> str:
    """Normalize a metro station name for cross-source matching.

    Strips parentheticals, common suffixes (站 / trailing 路 / · / -), and
    whitespace, then lowercases.  Identical station names from different
    sources (Amap / OSM / Wikidata / Wikipedia) collapse to the same key
    so that overlap detection and dedup produce consistent results.

    Note on ``路``: only the *trailing* ``路`` is stripped (e.g. "建国路"
    → "建国") so that stations with ``路`` in the middle of the name
    (e.g. "五路居", "十路口") are not corrupted.

    Examples:
        "西二旗站"         -> "西二旗"
        "西二旗(地铁)"     -> "西二旗"
        "西二旗（地铁）"   -> "西二旗"
        "建国路"           -> "建国"
        "五路居"           -> "五路居"   (preserved — 路 is not trailing)
        "海淀黄庄·换乘"    -> "海淀黄庄换乘"
        "Xierqi Station"  -> "xierqistation"
    """
    return _normalize_station_name(name)
