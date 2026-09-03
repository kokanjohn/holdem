"""
Baseball Savant (Statcast) expected-stats leaderboard — free, public,
unauthenticated CSV endpoint. Used as a cross-check on player rate
projections: a player running well above their expected stats (xwOBA <<
actual wOBA) is more likely to regress than one whose results match their
underlying quality of contact.
"""

import csv
import io

import requests

BASE = "https://baseballsavant.mlb.com/leaderboard"


def get_expected_stats(season, player_type="batter", min_pa=50):
    """player_type: 'batter' or 'pitcher'. Returns a dict keyed by MLBAM
    player_id with wOBA / expected-wOBA (and a few other fields) for the
    season, for use in apply_statcast_adjustment()."""
    url = f"{BASE}/expected_statistics"
    params = {
        "type": player_type,
        "year": season,
        "position": "",
        "team": "",
        "min": min_pa,
        "csv": "true",
    }
    resp = requests.get(url, params=params, timeout=20)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))

    out = {}
    for row in reader:
        pid = row.get("player_id")
        if not pid:
            continue
        try:
            entry = {
                "name": row.get("player_name"),
                "woba": float(row["woba"]) if row.get("woba") else None,
                "est_woba": float(row["est_woba"]) if row.get("est_woba") else None,
            }
        except (ValueError, KeyError):
            continue
        # Pitcher CSV uses the same wOBA/est_woba columns but they represent
        # wOBA *against* — field name kept consistent, meaning inferred from player_type.
        if player_type == "pitcher":
            entry["woba_against"] = entry.pop("woba")
            entry["est_woba_against"] = entry.pop("est_woba")
        out[int(pid)] = entry
    return out
