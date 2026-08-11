"""Quick test for the new housing_price_fetcher functions."""

from housing_price_fetcher import (
    CHROME_UA,
    KE_SUBDOMAIN,
    LIANJIA_SUBDOMAIN,
    _browser_launch_options,
    _filter_by_year,
    _is_blocked_url_title,
    _resolve_subdomain,
    diagnose_html,
    parse_chengjiao_page,
)

from urban_intervention.config.project import ACTIVE_CITIES


def test_resolve_subdomain_lianjia():
    sub, site = _resolve_subdomain("beijing")
    assert sub == "bj" and site == "lianjia"


def test_resolve_subdomain_beike_fallback():
    sub, site = _resolve_subdomain("urumqi")
    assert sub == "wlmq" and site == "beike"


def test_resolve_subdomain_no_coverage():
    sub, site = _resolve_subdomain("nonexistent")
    assert sub is None


def test_parse_empty_html():
    assert parse_chengjiao_page("") == []


def test_diagnose_captcha():
    html = "<html><title>CAPTCHA</title><body>captcha 验证 anti-spider</body></html>"
    diag = diagnose_html(html)
    assert diag["captcha"] > 0
    assert diag["anti-spider"] > 0


def test_blocked_url_title_detects_captcha_and_forbidden():
    assert _is_blocked_url_title("https://hip.lianjia.com/captcha?x=1", "CAPTCHA")
    assert _is_blocked_url_title("https://hip.lianjia.com/forbidden?id=1", "访问已被拦截")
    assert not _is_blocked_url_title("https://bj.lianjia.com/chengjiao/", "北京二手房成交")


def test_chrome_ua_is_recent():
    assert "Chrome/131" in CHROME_UA


def test_browser_launch_options_use_edge_when_available(monkeypatch):
    monkeypatch.setattr("housing_price_fetcher.EDGE_EXECUTABLE", r"C:\Edge\msedge.exe")
    opts = _browser_launch_options(login_mode=False)
    assert opts["executable_path"] == r"C:\Edge\msedge.exe"
    assert opts["headless"] is True


def test_all_44_cities_have_coverage():
    uncovered = [
        ck for ck in ACTIVE_CITIES if not LIANJIA_SUBDOMAIN.get(ck) and not KE_SUBDOMAIN.get(ck)
    ]
    assert uncovered == [], f"Cities without coverage: {uncovered}"


def test_filter_by_year():
    rows = [
        {"deal_year": 2010, "community": "old"},
        {"deal_year": 2020, "community": "new"},
        {"deal_year": None, "community": "unknown"},
    ]
    filtered = _filter_by_year(rows, min_year=2015)
    assert len(filtered) == 1
    assert filtered[0]["community"] == "new"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)
