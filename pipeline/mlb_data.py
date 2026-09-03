"""
Shared MLB Stats API helpers (statsapi.mlb.com — free, public, no key required).
Used by both the box-score scorer (scripts/nfbc_holdem_scoring.py) and the
projections pipeline (pipeline/*).
"""

import sys
import time
from datetime import datetime, timedelta

import requests

BASE = "https://statsapi.mlb.com/api/v1"


def get(path, params=None, retries=3, backoff=1.5):
    """GET against the MLB Stats API with basic retry/backoff."""
    url = f"{BASE}{path}" if path.startswith("/") else f"{BASE}/{path}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            if attempt == retries - 1:
                raise
            time.sleep(backoff ** attempt)


def ip_to_decimal_innings(ip_str):
    """MLB's IP notation counts thirds: '6.1' = 6 and 1/3 innings (1 out into the 7th),
    '6.2' = 6 and 2/3. Convert to true decimal innings."""
    if ip_str in (None, "", "0.0"):
        return 0.0
    whole, _, frac = str(ip_str).partition(".")
    whole = int(whole) if whole else 0
    frac = int(frac) if frac else 0
    return whole + frac / 3.0


def get_teams(season):
    """All 30 active MLB teams with division/league, for grouping standings
    without hardcoding team IDs."""
    data = get("/teams", params={"sportId": 1, "season": season, "activeStatus": "Y"})
    teams = []
    for t in data.get("teams", []):
        teams.append({
            "id": t["id"],
            "name": t["name"],
            "abbreviation": t.get("abbreviation"),
            "league": t.get("league", {}).get("name"),
            "leagueId": t.get("league", {}).get("id"),
            "division": t.get("division", {}).get("name"),
            "divisionId": t.get("division", {}).get("id"),
        })
    return teams


def get_team_runs_scored_allowed(season):
    """Runs scored and runs allowed per team for the season, for Pythagorean win%.
    Pulled from team-level aggregate hitting/pitching stats."""
    hitting = get("/teams/stats", params={
        "stats": "season", "group": "hitting", "season": season, "sportId": 1,
    })
    pitching = get("/teams/stats", params={
        "stats": "season", "group": "pitching", "season": season, "sportId": 1,
    })

    runs_scored = {}
    for split in hitting.get("stats", [{}])[0].get("splits", []):
        team_id = split["team"]["id"]
        runs_scored[team_id] = int(split["stat"].get("runs", 0))

    runs_allowed = {}
    for split in pitching.get("stats", [{}])[0].get("splits", []):
        team_id = split["team"]["id"]
        runs_allowed[team_id] = int(split["stat"].get("runs", 0))

    out = {}
    for team_id in set(runs_scored) | set(runs_allowed):
        out[team_id] = {
            "runsScored": runs_scored.get(team_id, 0),
            "runsAllowed": runs_allowed.get(team_id, 0),
        }
    return out


def get_team_hitting_totals(season):
    """Full team-level hitting aggregate for the season (used for league-average
    baselines in the player rate-shrinkage model)."""
    data = get("/teams/stats", params={
        "stats": "season", "group": "hitting", "season": season, "sportId": 1,
    })
    out = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        s = split["stat"]
        out[split["team"]["id"]] = {
            "plateAppearances": int(s.get("plateAppearances", 0)),
            "atBats": int(s.get("atBats", 0)),
            "hits": int(s.get("hits", 0)),
            "doubles": int(s.get("doubles", 0)),
            "triples": int(s.get("triples", 0)),
            "homeRuns": int(s.get("homeRuns", 0)),
            "rbi": int(s.get("rbi", 0)),
            "stolenBases": int(s.get("stolenBases", 0)),
            "baseOnBalls": int(s.get("baseOnBalls", 0)),
            "hitByPitch": int(s.get("hitByPitch", 0)),
            "runs": int(s.get("runs", 0)),
        }
    return out


def get_team_pitching_totals(season):
    """Full team-level pitching aggregate for the season."""
    data = get("/teams/stats", params={
        "stats": "season", "group": "pitching", "season": season, "sportId": 1,
    })
    out = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        s = split["stat"]
        out[split["team"]["id"]] = {
            "inningsPitched": ip_to_decimal_innings(s.get("inningsPitched", "0.0")),
            "earnedRuns": int(s.get("earnedRuns", 0)),
            "strikeOuts": int(s.get("strikeOuts", 0)),
            "wins": int(s.get("wins", 0)),
            "saves": int(s.get("saves", 0)),
            "gamesStarted": int(s.get("gamesStarted", 0)),
        }
    return out


def get_team_recent_game_count(team_id, season, last_n_days=30):
    """How many games this team has played in the trailing N days — the
    denominator for 'does this player start most games' role classification."""
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - timedelta(days=last_n_days)).strftime("%Y-%m-%d")
    data = get("/schedule", params={
        "sportId": 1, "teamId": team_id, "startDate": start, "endDate": end,
        "gameType": "R", "season": season,
    })
    count = 0
    for d in data.get("dates", []):
        for g in d.get("games", []):
            if g["status"]["detailedState"] in ("Final", "Game Over"):
                count += 1
    return count


def get_standings(season):
    """Current division standings + wild card ranks for both leagues."""
    data = get("/standings", params={
        "leagueId": "103,104", "season": season, "standingsTypes": "regularSeason",
    })
    rows = []
    for record in data.get("records", []):
        for t in record.get("teamRecords", []):
            rows.append({
                "teamId": t["team"]["id"],
                "teamName": t["team"]["name"],
                "wins": t.get("wins"),
                "losses": t.get("losses"),
                "gamesBack": t.get("gamesBack"),
                "wildCardGamesBack": t.get("wildCardGamesBack"),
                "divisionRank": t.get("divisionRank"),
                "wildCardRank": t.get("wildCardRank"),
                "clinched": t.get("clinched", False),
                "eliminationNumber": t.get("eliminationNumber"),
            })
    return rows


def get_postseason_schedule(season):
    """Full postseason schedule for a season (empty/sparse until the bracket
    is actually set)."""
    data = get("/schedule/postseason", params={"sportId": 1, "season": season})
    games = []
    for d in data.get("dates", []):
        for g in d.get("games", []):
            games.append({
                "date": d["date"],
                "gamePk": g["gamePk"],
                "series": g.get("seriesDescription"),
                "seriesGameNumber": g.get("seriesGameNumber"),
                "status": g["status"]["detailedState"],
                "awayId": g["teams"]["away"]["team"]["id"],
                "homeId": g["teams"]["home"]["team"]["id"],
                "away": g["teams"]["away"]["team"]["name"],
                "home": g["teams"]["home"]["team"]["name"],
            })
    return games


def get_player_fielding_games_by_position(player_id, season):
    """Games played at each defensive position this season, for computing
    NFBC-style position eligibility. Outfield sub-positions (LF/CF/RF) are
    summed into 'OF' since NFBC rosters only have a generic OF slot.

    Note: this queries the fielding stat group without a position filter,
    which the MLB Stats API returns as one split per position the player
    appeared at. This is standard behavior but hasn't been verified against
    a live call in this environment (no network access here) — if the
    response shape differs, this is the function to check first."""
    data = get(f"/people/{player_id}/stats", params={
        "stats": "season", "group": "fielding", "season": season,
    })
    out = {}
    for split in data.get("stats", [{}])[0].get("splits", []):
        pos = split.get("position", {}).get("abbreviation")
        games = int(split.get("stat", {}).get("games", 0))
        if not pos or games == 0:
            continue
        key = "OF" if pos in ("LF", "CF", "RF") else pos
        out[key] = out.get(key, 0) + games
    return out


def get_team_roster(team_id, roster_type="active"):
    """roster_type='active' returns only the current active (25/26-man) roster
    — players on the IL are excluded entirely, which silently hides them
    rather than flagging them. Use roster_type='40Man' from the EV engine so
    injured players stay visible and get explicitly flagged (see
    player_model.classify_availability) instead of just disappearing."""
    data = get(f"/teams/{team_id}/roster", params={"rosterType": roster_type})
    return [{
        "playerId": p["person"]["id"],
        "name": p["person"]["fullName"],
        "position": p.get("position", {}).get("abbreviation"),
        "status": p.get("status", {}).get("description"),
    } for p in data.get("roster", [])]


def get_player_season_stats(player_id, season, group):
    """group = 'hitting' or 'pitching'. Returns the season stat dict, or {} if none."""
    data = get(f"/people/{player_id}/stats", params={
        "stats": "season", "group": group, "season": season,
    })
    splits = data.get("stats", [{}])[0].get("splits", [])
    return splits[0]["stat"] if splits else {}


# Minor league levels served by the same public API, just a different sportId.
MINOR_LEAGUE_SPORT_IDS = {
    11: "AAA",
    12: "AA",
    13: "High-A",
    14: "Single-A",
    16: "Rookie",
}


def get_player_milb_season_stats_by_level(player_id, season, group):
    """group = 'hitting' or 'pitching'. Queries every MiLB level for this
    player/season and returns {level_name: stat_dict} for whichever levels
    they actually played at (a player promoted mid-season may appear at
    more than one level). Empty dict if the player has no MiLB record this
    season (e.g. a long-tenured MLB veteran) — cheap to check since most
    rostered players won't hit this path (see MLB_PA_THRESHOLD /
    MLB_IP_THRESHOLD gating in player_model.py, which only calls this for
    players with a thin MLB sample)."""
    out = {}
    for sport_id, level_name in MINOR_LEAGUE_SPORT_IDS.items():
        try:
            data = get(f"/people/{player_id}/stats", params={
                "stats": "season", "group": group, "season": season, "sportId": sport_id,
            })
        except requests.RequestException:
            continue
        splits = data.get("stats", [{}])[0].get("splits", [])
        if splits:
            out[level_name] = splits[0]["stat"]
    return out


def _get_full_gamelog(player_id, season, group):
    """Raw game log splits for the season, oldest first is NOT guaranteed by
    the API — callers should sort by date if order matters."""
    data = get(f"/people/{player_id}/stats", params={
        "stats": "gameLog", "group": group, "season": season,
    })
    return data.get("stats", [{}])[0].get("splits", [])


def get_player_recent_gamelog(player_id, season, group, last_n_days=30, splits=None):
    """group = 'hitting' or 'pitching'. Returns list of game log entries from
    the last N calendar days, for role/usage detection (everyday vs.
    platoon, rotation spot, closer vs. committee). Pass `splits` (from
    _get_full_gamelog) if you already fetched it for this player this run —
    avoids a duplicate API call."""
    if splits is None:
        splits = _get_full_gamelog(player_id, season, group)
    cutoff = datetime.utcnow() - timedelta(days=last_n_days)
    recent = []
    for s in splits:
        try:
            game_date = datetime.strptime(s.get("date", ""), "%Y-%m-%d")
        except ValueError:
            continue
        if game_date >= cutoff:
            recent.append(s)
    return recent


def get_player_last_n_games(player_id, season, group, n=15, splits=None):
    """group = 'hitting' or 'pitching'. Returns the player's last N games
    PLAYED (not a calendar-day window) — the right choice for a 'recent
    form' signal, since a calendar window catches wildly different numbers
    of appearances for an everyday hitter vs. a starting pitcher who only
    appears every ~5 games. Sorted oldest-to-newest. Pass `splits` if
    already fetched this run to avoid a duplicate call."""
    if splits is None:
        splits = _get_full_gamelog(player_id, season, group)
    dated = []
    for s in splits:
        try:
            d = datetime.strptime(s.get("date", ""), "%Y-%m-%d")
        except ValueError:
            continue
        dated.append((d, s))
    dated.sort(key=lambda x: x[0])
    return [s for _, s in dated[-n:]]
