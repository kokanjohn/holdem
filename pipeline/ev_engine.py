"""
Orchestration layer. Automatically detects how much of the bracket is real
vs. hypothetical and computes the best available projection for each stage:

- PRE-BRACKET (no postseason schedule yet): stores team power ratings +
  market World Series odds, and player rate/role projections. No round-by-
  round EV yet — there's no real bracket to project against.

- PARTIAL/FULL BRACKET (some series are real MLB-scheduled matchups):
  for a team's *current* round, simulates the real matchup (log5 blended
  with market h2h odds where available). For rounds beyond the currently
  known bracket, approximates using the team's strength against the
  average remaining-playoff-team quality — clearly flagged as a
  hypothetical estimate that sharpens once that round's real matchup
  exists.

Run this daily. Early in the postseason most of the value is in the
player rate/role numbers; the round-advancement numbers get progressively
more concrete (and more useful) as each round's real matchups lock in.
"""

import argparse
import sys
from datetime import datetime

from . import mlb_data, odds_client, team_model, player_model, savant_data, firestore_utils, position_eligibility

ROUND_INFO = {
    "Division Series": {"order": 1, "bestOf": 5, "multiplier": 1},
    "League Championship Series": {"order": 2, "bestOf": 7, "multiplier": 2},
    "World Series": {"order": 3, "bestOf": 7, "multiplier": 3},
}


def _round_key_from_series_description(series_description):
    """MLB's seriesDescription strings vary ('ALDS', 'NLDS', 'ALCS', 'NLCS',
    'World Series', etc.) — normalize to our ROUND_INFO keys."""
    if not series_description:
        return None
    s = series_description.lower()
    if "world series" in s:
        return "World Series"
    if "championship series" in s or "lcs" in s:
        return "League Championship Series"
    if "division series" in s or "ds" in s:
        return "Division Series"
    return None


def get_bracket_state(season):
    """Group the postseason schedule into distinct series (round + two teams),
    with each series' current status."""
    games = mlb_data.get_postseason_schedule(season)
    series = {}
    for g in games:
        round_key = _round_key_from_series_description(g.get("series"))
        if round_key is None:
            continue
        matchup_key = (round_key, frozenset([g["awayId"], g["homeId"]]))
        entry = series.setdefault(matchup_key, {
            "round": round_key,
            "teamIds": list(frozenset([g["awayId"], g["homeId"]])),
            "teamNames": [g["away"], g["home"]],
            "games": [],
        })
        entry["games"].append(g)
    return list(series.values())


def try_get_market_ws_odds():
    try:
        probs, remaining = odds_client.get_mlb_futures(market="outrights")
        print(f"  [odds] World Series market pulled, {remaining} credits remaining", file=sys.stderr)
        return probs
    except Exception as e:  # noqa: broad — this is a best-effort enrichment, never fatal
        print(f"  [odds] Skipping market odds: {e}", file=sys.stderr)
        return {}


def compute_team_power_ratings(season):
    teams = {t["id"]: t for t in mlb_data.get_teams(season)}
    runs = mlb_data.get_team_runs_scored_allowed(season)
    ratings = team_model.build_team_power_ratings(runs, teams)

    market_ws = try_get_market_ws_odds()
    for r in ratings:
        r["marketWSOdds"] = market_ws.get(r["teamName"])

    return ratings, teams, runs


def estimate_series(team_a_id, team_b_id, teams, runs, best_of, ratings_by_id):
    pythag_a = ratings_by_id[team_a_id]
    pythag_b = ratings_by_id[team_b_id]

    market_a = None
    try:
        market_a, _ = odds_client.get_series_h2h_odds(
            teams[team_a_id]["name"], teams[team_b_id]["name"]
        )
    except Exception as e:  # noqa: best-effort
        print(f"  [odds] No series odds for {teams[team_a_id]['name']} vs "
              f"{teams[team_b_id]['name']}: {e}", file=sys.stderr)

    result = team_model.compute_matchup_advancement(
        pythag_a, pythag_b, best_of, market_prob_a_wins_game=market_a,
    )
    return result


def compute_bracket_round_probabilities(season):
    """Returns {teamId: {'currentRound': str|None, 'advanceCurrentRoundProb': float|None}}
    for every team currently alive in a real, scheduled series."""
    ratings, teams, runs = compute_team_power_ratings(season)
    ratings_by_id = {r["teamId"]: r["pythagoreanWinPct"] for r in ratings}
    bracket = get_bracket_state(season)

    team_round_probs = {}
    for series in bracket:
        round_info = ROUND_INFO[series["round"]]
        team_a, team_b = series["teamIds"]
        # MLB's schedule endpoint can return placeholder entries for rounds
        # that haven't been determined yet (e.g. "TBD"/wild-card-winner
        # stand-ins) before the real bracket is set — these have team IDs
        # that aren't in our real 30-team ratings dict. Skip rather than
        # crash; a real matchup will replace the placeholder once it exists.
        if team_a not in ratings_by_id or team_b not in ratings_by_id:
            print(f"  [bracket] Skipping series with unresolved/placeholder team "
                  f"(ids {team_a}, {team_b}) — likely a TBD slot pre-bracket.", file=sys.stderr)
            continue
        result = estimate_series(team_a, team_b, teams, runs, round_info["bestOf"], ratings_by_id)
        team_round_probs[team_a] = {
            "currentRound": series["round"],
            "advanceCurrentRoundProb": result["seriesAdvanceProb"],
            "modelGameProb": result["modelGameProb"],
            "marketGameProb": result["marketGameProb"],
        }
        team_round_probs[team_b] = {
            "currentRound": series["round"],
            "advanceCurrentRoundProb": round(1 - result["seriesAdvanceProb"], 4),
            "modelGameProb": round(1 - result["modelGameProb"], 4),
            "marketGameProb": round(1 - result["marketGameProb"], 4) if result["marketGameProb"] is not None else None,
        }
    return team_round_probs, ratings_by_id, teams


def run(season, push_firestore=False, contenders_only=False, statcast=True):
    print(f"Computing team power ratings for {season}...", file=sys.stderr)
    ratings, teams, runs = compute_team_power_ratings(season)

    print("Checking postseason bracket state...", file=sys.stderr)
    team_round_probs, ratings_by_id, _ = compute_bracket_round_probabilities(season)

    if team_round_probs:
        print(f"  Bracket partially/fully set — {len(team_round_probs)} teams "
              f"in a real, scheduled series.", file=sys.stderr)
    else:
        print("  No postseason schedule yet — pre-bracket mode. Storing team "
              "ratings + player rate/role projections only.", file=sys.stderr)

    if push_firestore:
        n = firestore_utils.batch_upsert(
            "team_power_ratings", ratings, id_fn=lambda r: f"{season}_{r['teamId']}"
        )
        print(f"Pushed {n} team power ratings to Firestore.", file=sys.stderr)

        today = datetime.utcnow().strftime("%Y-%m-%d")
        team_history = [
            {"date": today, "teamId": r["teamId"], "teamName": r["teamName"],
             "pythagoreanWinPct": r["pythagoreanWinPct"], "marketWSOdds": r.get("marketWSOdds")}
            for r in ratings
        ]
        firestore_utils.batch_upsert(
            "team_rating_history", team_history, id_fn=lambda r: f"{r['date']}_{r['teamId']}"
        )

        if team_round_probs:
            records = [{"teamId": tid, "season": season, **v} for tid, v in team_round_probs.items()]
            n = firestore_utils.batch_upsert(
                "team_round_probabilities", records, id_fn=lambda r: f"{season}_{r['teamId']}"
            )
            print(f"Pushed {n} team round probabilities to Firestore.", file=sys.stderr)

    # Player rate/role projections
    league_pts_pa = player_model.league_avg_points_per_pa(season)
    league_pts_ip = player_model.league_avg_points_per_inning(season)
    print(f"League avg: {league_pts_pa:.3f} pts/PA, {league_pts_ip:.3f} pts/IP", file=sys.stderr)

    savant_batters, savant_pitchers = {}, {}
    if statcast:
        try:
            savant_batters = savant_data.get_expected_stats(season, "batter")
            savant_pitchers = savant_data.get_expected_stats(season, "pitcher")
            print(f"  [savant] {len(savant_batters)} batters, {len(savant_pitchers)} "
                  f"pitchers with expected-stats data", file=sys.stderr)
        except Exception as e:  # noqa: best-effort enrichment
            print(f"  [savant] Skipping Statcast adjustment: {e}", file=sys.stderr)

    candidate_teams = ratings
    if contenders_only:
        # crude filter: top half of teams by power rating, to cut runtime
        candidate_teams = ratings[: max(1, len(ratings) // 2)]

    player_records = []
    for team in candidate_teams:
        team_id = team["teamId"]
        print(f"  Projecting roster: {team['teamName']}...", file=sys.stderr)
        # 40-man, not 'active' — active-roster pulls silently exclude IL'd
        # players entirely. Pulling the fuller roster and explicitly flagging
        # availability (below) means an injured player still shows up, just
        # clearly marked, instead of just disappearing or being projected as
        # if healthy.
        roster = mlb_data.get_team_roster(team_id, roster_type="40Man")
        for p in roster:
            pos = p["position"]
            is_pitcher = pos in ("P", "SP", "RP")
            availability = player_model.classify_availability(p.get("status"))
            group = "pitching" if is_pitcher else "hitting"

            try:
                # Fetch the game log once, reuse for role classification AND
                # recent-form/momentum — halves the API calls per player
                # versus each pulling it independently.
                splits = mlb_data._get_full_gamelog(p["playerId"], season, group)

                if is_pitcher:
                    rate = player_model.project_pitcher_rate(
                        p["playerId"], team_id, season, league_pts_ip,
                        savant_row=savant_pitchers.get(p["playerId"]),
                    )
                    role = player_model.classify_pitcher_role(p["playerId"], team_id, season, splits=splits)
                    eligible_positions, pos_detail = [role["role"]], {}
                    recent_form = player_model.compute_recent_form_pitcher(p["playerId"], season, splits=splits)
                    momentum = player_model.compute_momentum(
                        recent_form["recentPointsPerIP"] if recent_form else None, rate.get("pointsPerIP")
                    )
                    save_signal = player_model.pick_save_signal(recent_form, rate.get("savesPerTeamWin"))
                else:
                    rate = player_model.project_hitter_rate(
                        p["playerId"], season, league_pts_pa,
                        savant_row=savant_batters.get(p["playerId"]),
                    )
                    role = player_model.classify_hitter_role(p["playerId"], team_id, season, splits=splits)
                    eligible_positions, pos_detail = position_eligibility.compute_eligible_positions(
                        p["playerId"], season, roster_listed_position=pos
                    )
                    recent_form = player_model.compute_recent_form_hitter(p["playerId"], season, splits=splits)
                    momentum = player_model.compute_momentum(
                        recent_form["recentPointsPerPA"] if recent_form else None, rate.get("pointsPerPA")
                    )
                    save_signal = None
            except Exception as e:  # noqa: don't let one bad player stats call kill the run
                print(f"    [skip] {p['name']}: {e}", file=sys.stderr)
                continue

            if not availability["available"]:
                role["role"] = "IL"  # overrides the usage-derived role — can't be "everyday" while hurt

            record = {
                "season": season, "teamId": team_id, "teamName": team["teamName"],
                "playerId": p["playerId"], "player": p["name"],
                "rosterPosition": pos,  # raw MLB.com-style single position, kept for reference only
                "eligiblePositions": eligible_positions,
                "eligiblePositionsDisplay": "/".join(eligible_positions),
                "positionEligibilityFallback": pos_detail.get("fallbackUsed", False),
                "available": availability["available"],
                "statusNote": availability["status"],
                "momentum": momentum,  # real recent-vs-projected delta — this is what should drive trend arrows
                **(recent_form or {}),
                **rate, **role,
            }
            if save_signal:
                record.update(save_signal)
            round_prob = team_round_probs.get(team_id)
            if round_prob:
                record["currentRound"] = round_prob["currentRound"]
                record["advanceCurrentRoundProb"] = round_prob["advanceCurrentRoundProb"]
                record["roundMultiplier"] = ROUND_INFO[round_prob["currentRound"]]["multiplier"]
            player_records.append(record)

    if push_firestore and player_records:
        n = firestore_utils.batch_upsert(
            "player_projections", player_records,
            id_fn=lambda r: f"{season}_{r['playerId']}",
        )
        print(f"Pushed {n} player projections to Firestore.", file=sys.stderr)

        # Dated snapshot alongside the overwritten "current" doc above — this
        # is what makes real trend charts possible. Without this, Firestore
        # only ever holds today's numbers and there's no history to plot.
        today = datetime.utcnow().strftime("%Y-%m-%d")
        history_records = [
            {"date": today, "playerId": r["playerId"], "player": r["player"],
             "value": r.get("pointsPerPA", r.get("pointsPerIP")), "momentum": r.get("momentum")}
            for r in player_records
        ]
        n = firestore_utils.batch_upsert(
            "player_projection_history", history_records,
            id_fn=lambda r: f"{r['date']}_{r['playerId']}",
        )
        print(f"Pushed {n} player history snapshots to Firestore.", file=sys.stderr)

    if push_firestore:
        db = firestore_utils.get_firestore_client()
        from firebase_admin import firestore as _firestore
        db.collection("system_meta").document("last_run").set({
            "ranAt": _firestore.SERVER_TIMESTAMP,
            "season": season,
            "teamsProjected": len(candidate_teams),
            "playersProjected": len(player_records),
            "bracketState": "set" if team_round_probs else "pre-bracket",
        })
        print("Updated system_meta/last_run.", file=sys.stderr)

    return ratings, team_round_probs, player_records


def main():
    parser = argparse.ArgumentParser(description="NFBC Postseason Holdem projection pipeline")
    parser.add_argument("--season", required=True, type=int)
    parser.add_argument("--push-firestore", action="store_true")
    parser.add_argument("--contenders-only", action="store_true",
                         help="Only project the top half of teams by power rating (faster run)")
    parser.add_argument("--no-statcast", action="store_true", help="Skip the Statcast adjustment")
    args = parser.parse_args()

    ratings, team_round_probs, player_records = run(
        args.season, push_firestore=args.push_firestore,
        contenders_only=args.contenders_only, statcast=not args.no_statcast,
    )

    print("\n=== Top 10 Team Power Ratings ===")
    for r in ratings[:10]:
        print(f"{r['teamName']:25s} pythag={r['pythagoreanWinPct']:.3f}  "
              f"marketWS={r.get('marketWSOdds')}")

    if player_records:
        top = sorted(player_records, key=lambda r: r.get("pointsPerPA", r.get("pointsPerIP", 0)), reverse=True)
        print("\n=== Sample player projections ===")
        for r in top[:15]:
            print(r)


if __name__ == "__main__":
    main()
