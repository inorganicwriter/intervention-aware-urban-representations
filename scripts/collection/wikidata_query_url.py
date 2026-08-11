"""Export the Wikidata SPARQL query as a ready-to-paste URL for the browser.

Chrome respects Clash system proxy, so this is the most reliable way to
run the Wikidata query from behind the GFW. Steps for the user:
  1. Ensure Clash Global mode is ON
  2. Open Chrome
  3. Paste the URL printed by this script
  4. Wait for results → click "Download" → choose "JSON" format
  5. Save the file to data/archive/raw/transit/wikidata/wikidata_metro_raw.json
  6. Run: python scripts/collection/wikidata_transit_fetch.py --from-file data/archive/raw/transit/wikidata/wikidata_metro_raw.json
"""

import urllib.parse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Keep this query identical to the one in wikidata_transit_fetch.py so
# that --from-file produces the same schema.  The previous version had a
# redundant second P131 OPTIONAL bound to ?city_en (which always equaled
# ?city); we now use rdfs:label for the English label if needed.
SPARQL = """
SELECT ?station ?stationLabel ?geo ?opening ?line ?lineLabel ?city ?cityLabel WHERE {
  ?station wdt:P17 wd:Q148 ;
           wdt:P625 ?geo ;
           wdt:P81 ?line .
  OPTIONAL { ?station wdt:P1619 ?opening . }
  OPTIONAL { ?station wdt:P131 ?city . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "zh,en". }
}
LIMIT 8000
"""


def main():
    SPARQL_URL = "https://query.wikidata.org/sparql?format=json&query="
    full_url = SPARQL_URL + urllib.parse.quote(SPARQL.strip(), safe="")

    print("=" * 60)
    print("Open this URL in Chrome (with Clash Global ON):")
    print()
    print(full_url)
    print()
    print("=" * 60)
    print()
    print("After loading:")
    print("  1. Wait for results (may take 10-30s)")
    print("  2. Click 'Download' button")
    print("  3. Choose 'JSON' format")
    print(f"  4. Save to: {BASE_DIR / 'data' / 'external' / 'wikidata_metro_raw.json'}")
    print()
    print("Then run:")
    print(
        "  python scripts/collection/wikidata_transit_fetch.py --from-file data/archive/raw/transit/wikidata/wikidata_metro_raw.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
