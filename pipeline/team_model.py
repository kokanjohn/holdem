"""
Team advancement model.

PRE-BRACKET (now through the end of the Wild Card round):
    No real DS/LCS/WS matchups exist yet, so there's nothing concrete to
    simulate. Instead we compute a power rating (Pythagorean win%) per team
    and pull market World Series odds — both stored for research/tracking,
    not decomposed into round-by-round probabilities (that would require
    guessing a bracket that doesn't exist yet).

POST-BRACKET (once real series matchups are known):
    For a given matchup, simulate the series using the log5 method on each
    team's Pythagorean win%, blended with real de-vigged sportsbook h2h odds
    when available (logit-averaged — treats model and market as equally
    informative by default). Monte Carlo the series many times to get a
    round-win probability, then chain across rounds to get
    P(team active in round 2) / P(team active in round 3) — the numbers that
    directly scale the multiplier in the EV engine.
"""

import math
import random

PYTHAG_EXPONENT = 1.83


def pythagorean_win_pct(runs_scored, runs_allowed, exponent=PYTHAG_EXPONENT):
    if runs_scored == 0 and runs_allowed == 0:
        return 0.5
    rs_e = runs_scored ** exponent
    ra_e = runs_allowed ** exponent
    return rs_e / (rs_e + ra_e)


def log5(pct_a, pct_b):
    """Probability team A beats team B in a single game, given each team's
    true winning percentage (log5 / Bill James method)."""
    pct_a = min(max(pct_a, 0.001), 0.999)
    pct_b = min(max(pct_b, 0.001), 0.999)
    numerator = pct_a - pct_a * pct_b
    denominator = pct_a + pct_b - 2 * pct_a * pct_b
    return numerator / denominator


def _logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))


def blend_probabilities(model_prob, market_prob, model_weight=0.5):
    """Average two probabilities in logit space (more statistically sound
    than a raw arithmetic average). If market_prob is None, returns
    model_prob unchanged."""
    if market_prob is None:
        return model_prob
    blended_logit = model_weight * _logit(model_prob) + (1 - model_weight) * _logit(market_prob)
    return _sigmoid(blended_logit)


def simulate_series(prob_a_wins_game, best_of, n_sims=10000, seed=None):
    """Monte Carlo simulate a best-of-N series given team A's per-game win
    probability. Returns P(team A wins the series)."""
    rng = random.Random(seed)
    wins_needed = best_of // 2 + 1
    a_series_wins = 0
    for _ in range(n_sims):
        a_wins = 0
        b_wins = 0
        while a_wins < wins_needed and b_wins < wins_needed:
            if rng.random() < prob_a_wins_game:
                a_wins += 1
            else:
                b_wins += 1
        if a_wins == wins_needed:
            a_series_wins += 1
    return a_series_wins / n_sims


def compute_matchup_advancement(
    team_a_pythag, team_b_pythag, best_of,
    market_prob_a_wins_game=None, model_weight=0.5, n_sims=10000, seed=None,
):
    """Full pipeline for one series: log5 model prob -> blend with market ->
    Monte Carlo the series -> P(team A advances)."""
    model_prob_a_wins_game = log5(team_a_pythag, team_b_pythag)
    blended_game_prob = blend_probabilities(
        model_prob_a_wins_game, market_prob_a_wins_game, model_weight=model_weight
    )
    prob_a_advances = simulate_series(blended_game_prob, best_of, n_sims=n_sims, seed=seed)
    return {
        "modelGameProb": round(model_prob_a_wins_game, 4),
        "marketGameProb": round(market_prob_a_wins_game, 4) if market_prob_a_wins_game else None,
        "blendedGameProb": round(blended_game_prob, 4),
        "seriesAdvanceProb": round(prob_a_advances, 4),
    }


def build_team_power_ratings(runs_by_team, team_lookup):
    """runs_by_team: {teamId: {'runsScored':.., 'runsAllowed':..}}
    team_lookup: {teamId: {'name':.., 'division':.., 'league':..}}
    Returns list of dicts with Pythagorean win% per team, for the
    pre-bracket research view."""
    ratings = []
    for team_id, runs in runs_by_team.items():
        info = team_lookup.get(team_id, {})
        pct = pythagorean_win_pct(runs["runsScored"], runs["runsAllowed"])
        ratings.append({
            "teamId": team_id,
            "teamName": info.get("name"),
            "division": info.get("division"),
            "league": info.get("league"),
            "runsScored": runs["runsScored"],
            "runsAllowed": runs["runsAllowed"],
            "pythagoreanWinPct": round(pct, 4),
        })
    ratings.sort(key=lambda r: r["pythagoreanWinPct"], reverse=True)
    return ratings
