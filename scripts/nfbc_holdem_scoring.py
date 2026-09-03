"""
NFBC Postseason Holdem — Daily Stat Puller & Scorer (Firestore-enabled)
========================================================================

Pulls MLB postseason box scores from the free, public MLB Stats API
(statsapi.mlb.com — no API key required) and converts each player's
game-by-game stat line into NFBC Postseason Holdem fantasy points.

This version adds:
  - a stable numeric playerId on every row (used as part of the Firestore
    document ID, so re-running a date safely overwrites/updates rather
    than duplicating rows — important since box scores can get revised
    after official scoring reviews)
  - an optional --push-firestore flag for use in the GitHub Actions
    daily automation

NOTE: This script does NOT apply the round multiplier (1x/2x/3x) or
enforce roster rules — that depends on which players YOU held on your
team each round, not on the raw data. Layer that on top separately
once you have your drafted roster.

Usage:
    python nfbc_holdem_scoring.py --date 2026-10-03
    python nfbc_holdem_scoring.py --start 2026-10-03 --end 2026-10-05
    python nfbc_holdem_scoring.py --postseason-schedule 2026
    python nfbc_holdem_scoring.py --date 2026-10-03 --push-firestore

Requires: requests, pandas
    pip install requests pandas --break-system-packages
Optional (only for --push-firestore): firebase-admin
    pip install firebase-admin --break-system-packages
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import requests
import pandas as pd

BASE = "https://statsapi.mlb.com/api/v1"

# ---- NFBC Postseason Holdem scoring table ----
HIT_PTS = {
    "runs": 1.00,
    "singles": 1.00,      # derived: H - 2B - 3B - HR
    "doubles": 2.00,
    "triples": 3.00,
    "homeRuns": 4.00,
    "rbi": 1.00,
    "stolenBases": 1.00,
    "baseOnBalls": 1.00,
    "hitByPitch": 1.00,
    "outs": -0.25,         # derived: AB - H
}

PITCH_PTS = {
    "inningsPitched": 1.00,   # per full inning; API gives IP as e.g. 6.1 = 6 1/3 innings
    "earnedRuns": -1.00,
    "strikeOuts": 1.00,
    "wins": 4.00,
    "saves": 4.00,
}


def _get(url, params=None, retries=3, backoff=1.5):
    """GET with basic retry/backoff — the API is reliable but be polite/robust."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(backoff ** attempt)


def ip_to_outs_innings(ip_str):
    """Convert MLB's IP notation (e.g. '6.1' = 6 and 1/3 innings) to true decimal innings
    for point purposes. MLB counts IP in thirds: .1 = 1 out, .2 = 2 outs."""
    if ip_str in (None, "", "0.0"):
        return 0.0
    whole, _, frac = str(ip_str).partition(".")
    whole = int(whole) if whole else 0
    frac = int(frac) if frac else 0
    return whole + frac / 3.0


def get_schedule_game_pks(date_str):
    data = _get(f"{BASE}/schedule", params={"sportId": 1, "date": date_str})
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            games.append({
                "gamePk": g["gamePk"],
                "status": g["status"]["detailedState"],
                "away": g["teams"]["away"]["team"]["name"],
                "home": g["teams"]["home"]["team"]["name"],
                "gameType": g.get("gameType"),
            })
    return games


def get_postseason_schedule(season):
    data = _get(f"{BASE}/schedule/postseason", params={"sportId": 1, "season": season})
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            games.append({
                "date": d["date"],
                "gamePk": g["gamePk"],
                "series": g.get("seriesDescription"),
                "status": g["status"]["detailedState"],
                "away": g["teams"]["away"]["team"]["name"],
                "home": g["teams"]["home"]["team"]["name"],
            })
    return games


def get_boxscore_player_lines(game_pk):
    data = _get(f"{BASE}/game/{game_pk}/boxscore")
    rows = []
    for side in ("away", "home"):
        team_info = data["teams"][side]
        team_name = team_info["team"]["name"]
        players = team_info.get("players", {})
        for _pid, pdata in players.items():
            person = pdata.get("person", {})
            name = person.get("fullName")
            player_id = person.get("id")
            position = pdata.get("position", {}).get("abbreviation")

            batting = pdata.get("stats", {}).get("batting", {})
            pitching = pdata.get("stats", {}).get("pitching", {})

            has_bat = batting.get("atBats", 0) or batting.get("plateAppearances", 0)
            has_pitch = pitching.get("inningsPitched") not in (None, "0.0")

            if not has_bat and not has_pitch:
                continue

            row = {
                "gamePk": game_pk,
                "team": team_name,
                "playerId": player_id,
                "player": name,
                "position": position,
            }

            if has_bat:
                h = batting.get("hits", 0)
                doubles = batting.get("doubles", 0)
                triples = batting.get("triples", 0)
                hr = batting.get("homeRuns", 0)
                ab = batting.get("atBats", 0)
                singles = h - doubles - triples - hr
                outs = ab - h
                row.update({
                    "AB": ab, "R": batting.get("runs", 0), "H": h,
                    "1B": singles, "2B": doubles, "3B": triples, "HR": hr,
                    "RBI": batting.get("rbi", 0), "SB": batting.get("stolenBases", 0),
                    "BB": batting.get("baseOnBalls", 0), "HBP": batting.get("hitByPitch", 0),
                    "Outs": outs,
                })
                bat_pts = (
                    row["R"] * HIT_PTS["runs"] +
                    row["1B"] * HIT_PTS["singles"] +
                    row["2B"] * HIT_PTS["doubles"] +
                    row["3B"] * HIT_PTS["triples"] +
                    row["HR"] * HIT_PTS["homeRuns"] +
                    row["RBI"] * HIT_PTS["rbi"] +
                    row["SB"] * HIT_PTS["stolenBases"] +
                    row["BB"] * HIT_PTS["baseOnBalls"] +
                    row["HBP"] * HIT_PTS["hitByPitch"] +
                    row["Outs"] * HIT_PTS["outs"]
                )
                row["battingPts"] = round(bat_pts, 2)
            else:
                row["battingPts"] = 0.0

            if has_pitch:
                ip_decimal = ip_to_outs_innings(pitching.get("inningsPitched"))
                row.update({
                    "IP": pitching.get("inningsPitched", "0.0"),
                    "ER": pitching.get("earnedRuns", 0),
                    "K": pitching.get("strikeOuts", 0),
                    "W": pitching.get("wins", 0),
                    "SV": pitching.get("saves", 0),
                })
                pitch_pts = (
                    ip_decimal * PITCH_PTS["inningsPitched"] +
                    row["ER"] * PITCH_PTS["earnedRuns"] +
                    row["K"] * PITCH_PTS["strikeOuts"] +
                    row["W"] * PITCH_PTS["wins"] +
                    row["SV"] * PITCH_PTS["saves"]
                )
                row["pitchingPts"] = round(pitch_pts, 2)
            else:
                row["pitchingPts"] = 0.0

            row["totalPts"] = round(row["battingPts"] + row["pitchingPts"], 2)
            rows.append(row)
    return rows


def score_date(date_str, only_final=True):
    games = get_schedule_game_pks(date_str)
    all_rows = []
    for g in games:
        if only_final and g["status"] not in ("Final", "Game Over"):
            print(f"  [skip] gamePk {g['gamePk']} ({g['away']} @ {g['home']}) "
                  f"status={g['status']} — not final yet", file=sys.stderr)
            continue
        rows = get_boxscore_player_lines(g["gamePk"])
        for r in rows:
            r["date"] = date_str
        all_rows.extend(rows)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows)
    cols = ["date", "gamePk", "team", "playerId", "player", "position",
            "AB", "R", "H", "1B", "2B", "3B", "HR", "RBI", "SB", "BB", "HBP", "Outs", "battingPts",
            "IP", "ER", "K", "W", "SV", "pitchingPts", "totalPts"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values("totalPts", ascending=False)


def score_date_range(start_str, end_str, only_final=True):
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    frames = []
    d = start
    while d <= end:
        ds = d.strftime("%Y-%m-%d")
        print(f"Pulling {ds} ...", file=sys.stderr)
        df = score_date(ds, only_final=only_final)
        if not df.empty:
            frames.append(df)
        d += timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def push_to_firestore(df, collection="player_game_stats"):
    """Upsert every row into Firestore. Doc ID = date_gamePk_playerId, so
    reruns update in place instead of creating duplicates."""
    import firebase_admin
    from firebase_admin import credentials, firestore

    if not firebase_admin._apps:
        cred_json = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
        if not cred_json:
            raise RuntimeError(
                "FIREBASE_SERVICE_ACCOUNT_JSON env var not set. "
                "Paste the full Firebase service account JSON into that env var / GitHub secret."
            )
        cred_dict = json.loads(cred_json)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)

    db = firestore.client()
    batch = db.batch()
    count = 0
    total = 0
    for _, row in df.iterrows():
        pid = row.get("playerId")
        pid_part = str(int(pid)) if pd.notna(pid) else "unknown"
        doc_id = f"{row['date']}_{row['gamePk']}_{pid_part}"
        doc_ref = db.collection(collection).document(doc_id)
        data = row.to_dict()
        # Firestore doesn't like NaN — clean it up
        data = {k: (None if isinstance(v, float) and pd.isna(v) else v) for k, v in data.items()}
        batch.set(doc_ref, data, merge=True)
        count += 1
        total += 1
        if count == 400:  # stay under Firestore's 500-write batch limit
            batch.commit()
            batch = db.batch()
            count = 0
    if count > 0:
        batch.commit()

    # Log the run for visibility/debugging
    db.collection("pull_runs").add({
        "ranAt": firestore.SERVER_TIMESTAMP,
        "rowsWritten": total,
        "dates": sorted(df["date"].unique().tolist()),
    })
    print(f"Pushed {total} rows to Firestore collection '{collection}'.", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="NFBC Postseason Holdem stat/scoring puller")
    parser.add_argument("--date", help="Single date YYYY-MM-DD")
    parser.add_argument("--start", help="Range start YYYY-MM-DD")
    parser.add_argument("--end", help="Range end YYYY-MM-DD")
    parser.add_argument("--postseason-schedule", metavar="SEASON",
                         help="List full postseason schedule for a season, e.g. 2026")
    parser.add_argument("--include-unfinished", action="store_true",
                         help="Include in-progress games too (partial stats)")
    parser.add_argument("--out", default=None, help="Optional CSV output path")
    parser.add_argument("--push-firestore", action="store_true",
                         help="Push results to Firestore (requires FIREBASE_SERVICE_ACCOUNT_JSON env var)")
    args = parser.parse_args()

    if args.postseason_schedule:
        games = get_postseason_schedule(args.postseason_schedule)
        df = pd.DataFrame(games)
        print(df.to_string(index=False))
        if args.out:
            df.to_csv(args.out, index=False)
        return

    only_final = not args.include_unfinished

    if args.date:
        df = score_date(args.date, only_final=only_final)
    elif args.start and args.end:
        df = score_date_range(args.start, args.end, only_final=only_final)
    else:
        parser.error("Provide --date, or --start/--end, or --postseason-schedule")

    if df.empty:
        print("No completed games / player stat lines found for that range.")
        return

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nSaved to {args.out}", file=sys.stderr)

    if args.push_firestore:
        push_to_firestore(df)


if __name__ == "__main__":
    main()
