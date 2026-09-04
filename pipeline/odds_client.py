"""
Client for The Odds API (the-odds-api.com) — free tier: 500 credits/month,
no card required. Pulling MLB futures once/day during the postseason uses a
small fraction of that.

Requires env var ODDS_API_KEY (set as a GitHub secret in automation; never
hardcode the key in source).
"""

import os
import requests

BASE = "https://api.the-odds-api.com/v4"


class OddsAPIError(RuntimeError):
    pass


def _get_key():
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        raise OddsAPIError(
            "ODDS_API_KEY env var not set. Add it as a GitHub secret / local "
            "env var — never hardcode an API key in source."
        )
    return key


def american_to_implied_prob(american_odds):
    """Convert American odds to raw (vig-included) implied probability."""
    o = float(american_odds)
    if o > 0:
        return 100.0 / (o + 100.0)
    return -o / (-o + 100.0)


def devig(prob_dict):
    """Normalize a dict of {outcome: raw_implied_prob} so probabilities sum to 1,
    removing the sportsbook's overround (vig)."""
    total = sum(prob_dict.values())
    if total <= 0:
        return {k: 0.0 for k in prob_dict}
    return {k: v / total for k, v in prob_dict.items()}


def get_mlb_futures(market="outrights"):
    """Pull MLB World Series (or pennant) outright odds for all teams,
    de-vigged into probabilities.

    market: 'outrights' for World Series winner. Some books also expose
    'alternate' markets like league pennant winners — check
    GET {BASE}/sports/baseball_mlb/odds?... markets for what's currently live;
    availability varies book to book and by time of season.

    NOTE: querying markets='outrights' on the main 'baseball_mlb' sport key
    returned a 422 in testing. Best available guess (UNVERIFIED — I don't
    have network access to confirm live): The Odds API likely treats season-
    long futures as a separate sport key rather than a market on the regular
    sport, following the pattern seen elsewhere in their catalog (e.g.
    'americanfootball_nfl_super_bowl_winner'). This function now tries the
    regular sport first, then falls back to a guessed futures-specific sport
    key. If both fail, check https://api.the-odds-api.com/v4/sports/?apiKey=...
    (with your real key) to see the exact list of valid sport keys — that's
    the authoritative source, not this comment.
    """
    key = _get_key()
    attempts = [
        ("baseball_mlb", market),
        ("baseball_mlb_world_series_winner", "outrights"),
    ]
    last_error = None
    for sport_key, market_key in attempts:
        try:
            resp = requests.get(
                f"{BASE}/sports/{sport_key}/odds",
                params={
                    "apiKey": key,
                    "regions": "us",
                    "markets": market_key,
                    "oddsFormat": "american",
                },
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException as e:
            last_error = e
            data = None
    if data is None:
        raise OddsAPIError(
            f"All sport-key attempts for MLB futures failed. Last error: {last_error}. "
            f"Check {BASE}/sports/?apiKey=YOUR_KEY for the real list of valid sport keys."
        )

    # Average implied probability across all books offering the market, then de-vig.
    team_odds = {}  # team_name -> list of raw implied probs across books
    for event in data:
        for bookmaker in event.get("bookmakers", []):
            for m in bookmaker.get("markets", []):
                if m.get("key") != market_key:
                    continue
                for outcome in m.get("outcomes", []):
                    team = outcome["name"]
                    prob = american_to_implied_prob(outcome["price"])
                    team_odds.setdefault(team, []).append(prob)

    avg_probs = {team: sum(vals) / len(vals) for team, vals in team_odds.items()}
    return devig(avg_probs), resp.headers.get("x-requests-remaining")


def get_series_h2h_odds(team_a_name, team_b_name):
    """Best-effort pull of a specific postseason series' moneyline/h2h odds,
    once that series exists as a bettable event (i.e. after the bracket is
    set — books don't post series-specific lines earlier than that).

    Returns (prob_a_wins_game, prob_b_wins_game) per-game implied win
    probabilities, de-vigged, or (None, None) if no matching event is found.
    """
    key = _get_key()
    resp = requests.get(
        f"{BASE}/sports/baseball_mlb/odds",
        params={
            "apiKey": key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
        },
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()

    for event in data:
        teams = {event.get("home_team"), event.get("away_team")}
        if {team_a_name, team_b_name} <= teams or teams & {team_a_name, team_b_name}:
            probs = {}
            for bookmaker in event.get("bookmakers", []):
                for m in bookmaker.get("markets", []):
                    if m.get("key") != "h2h":
                        continue
                    for outcome in m.get("outcomes", []):
                        probs.setdefault(outcome["name"], []).append(
                            american_to_implied_prob(outcome["price"])
                        )
            if probs:
                avg = {k: sum(v) / len(v) for k, v in probs.items()}
                devigged = devig(avg)
                return (
                    devigged.get(team_a_name),
                    devigged.get(team_b_name),
                )
    return None, None
