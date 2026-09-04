"""
Player projection model: two independent pieces.

1. RATE projection — "how many fantasy points per PA / per inning is this
   player's true talent, not just their season-to-date results?"
   Season totals are regressed toward the league-average rate, weighted by
   sample size (classic Empirical-Bayes shrinkage: small samples get pulled
   hard toward league average, large samples are trusted more). Optionally
   cross-checked against Statcast expected stats (xwOBA vs actual wOBA) to
   catch players over/underperforming their underlying skill.

   Call-ups with a thin MLB sample (below MLB_PA_THRESHOLD / MLB_IP_THRESHOLD)
   don't just shrink to flat league average — the model first pulls their
   MiLB record across every level played this season, translates it toward
   MLB quality of competition (rough historical MLE-style factors, weighted
   by level with lower levels trusted far less), and shrinks THAT into a
   MiLB-informed prior before blending in the small MLB sample. A hot AAA
   prospect and a replacement-level scrub who both have 20 MLB PA will not
   get the same projection.

2. ROLE projection — "how much playing time is this player actually going
   to get?" Everyday player vs. platoon vs. bench; rotation spot vs.
   reliever vs. closer. Derived from trailing 30-day usage, since that's
   the best available signal for current role (season-long role can be
   stale after trades, injuries, demotions, etc.). Note: a player called up
   in just the last few days will show a small/uncertain sample here too —
   that's an honest reflection of real uncertainty, not a bug to paper over.

3. AVAILABILITY — separate from rate/role: classify_availability() reads
   the roster status string (Active, 10-Day-IL, Restricted List, etc.) and
   flags a player as unavailable rather than silently dropping him from the
   pool or projecting him as if healthy. Skill rate doesn't change because
   someone's hurt, but he shouldn't show up as an undifferentiated target
   either — see ev_engine.py for how this gets attached to each record.

4. RECENT FORM / MOMENTUM — compute_recent_form_hitter/pitcher() compares a
   player's rate over their actual last N games played (not a calendar
   window) against their season-long projection. This is real, computed
   trend data — not a placeholder — and generalizes the same trailing-
   window idea used for closer detection to every player. Same game-log
   fetch is reused for role classification via the optional `splits` param,
   so this doesn't double the number of API calls per player.

Both feed the EV engine, which multiplies rate x expected volume x team
survival probability x round multiplier.
"""

from . import mlb_data, scoring_rules

HITTER_SHRINKAGE_K_PA = 200      # PA at which player rate and league rate are weighted equally
PITCHER_SHRINKAGE_K_IP = 40      # innings at which player rate and league rate are weighted equally
STATCAST_ADJUSTMENT_CAP = 0.15   # cap the xwOBA/wOBA nudge at +/-15% to avoid overcorrection

# --- Minor-league call-up handling -----------------------------------------
# Below this much MLB sample, pull MiLB stats and build a MiLB-informed prior
# instead of shrinking straight to the flat league average. A September
# call-up with 20 MLB PA but a huge AAA track record should NOT project the
# same as a replacement-level scrub with 20 MLB PA and nothing behind it.
MLB_PA_THRESHOLD = 130
MLB_IP_THRESHOLD = 20

# Production discount applied to MiLB rate stats when translating to MLB
# quality of competition ("Major League Equivalency" factors). These are
# rough historical approximations from the sabermetric literature (Bill
# James' original MLEs, Clay Davenport translations, and later work) — NOT
# empirically re-derived here. Treat as tunable, not gospel.
LEVEL_TRANSLATION_FACTOR = {
    "AAA": 0.88,
    "AA": 0.72,
    "High-A": 0.55,
    "Single-A": 0.40,
    "Rookie": 0.30,
}

# How much a given level's sample size should count relative to an equal
# amount of real MLB sample, when using it as shrinkage weight. Lower than
# the translation factor itself — published research (e.g. Baseball
# Prospectus's MLE work) finds translation accuracy degrades sharply below
# AAA, so even a lot of Single-A innings shouldn't move the projection much.
LEVEL_CONFIDENCE = {
    "AAA": 0.60,
    "AA": 0.40,
    "High-A": 0.25,
    "Single-A": 0.15,
    "Rookie": 0.10,
}

MILB_PRIOR_K_PA = 150   # how hard the MiLB-informed prior itself shrinks toward flat league average
MILB_PRIOR_K_IP = 30

# --- Injury / availability handling -----------------------------------------
# Roster status descriptions that mean "not currently playable" — matched as
# case-insensitive substrings against whatever MLB Stats API returns (it uses
# strings like "10-Day-IL", "60-Day-IL", "Bereavement List", "Paternity List",
# "Restricted List", "Suspended List"). This governs AVAILABILITY, not rate —
# a player's skill projection doesn't change because he's hurt, but he
# shouldn't be surfaced as a target without a visible flag either.
INACTIVE_STATUS_KEYWORDS = ["IL", "Injured", "Bereavement", "Paternity", "Restricted", "Suspended"]


def classify_availability(status_description):
    """status_description comes straight from mlb_data.get_team_roster()'s
    'status' field. Returns {'available': bool, 'status': str|None}."""
    if not status_description:
        return {"available": True, "status": status_description}
    is_inactive = any(k.lower() in status_description.lower() for k in INACTIVE_STATUS_KEYWORDS)
    return {"available": not is_inactive, "status": status_description}


def shrink(observed_rate, sample_size, prior_rate, k):
    """Empirical-Bayes-style shrinkage: blend an observed rate with a prior
    (league average), weighted by how much sample size backs the observed
    rate versus the shrinkage constant k."""
    if sample_size + k == 0:
        return prior_rate
    return (sample_size * observed_rate + k * prior_rate) / (sample_size + k)


def league_avg_points_per_pa(season):
    totals = mlb_data.get_team_hitting_totals(season)
    total_pa = sum(t["plateAppearances"] for t in totals.values())
    total_pts = sum(
        scoring_rules.batting_points_from_totals(
            ab=t["atBats"], h=t["hits"], doubles=t["doubles"], triples=t["triples"],
            hr=t["homeRuns"], rbi=t["rbi"], sb=t["stolenBases"], bb=t["baseOnBalls"],
            hbp=t["hitByPitch"], runs=t["runs"],
        ) for t in totals.values()
    )
    return total_pts / total_pa if total_pa else 0.0


def league_avg_points_per_inning(season):
    totals = mlb_data.get_team_pitching_totals(season)
    total_ip = sum(t["inningsPitched"] for t in totals.values())
    # Wins/saves excluded here — they're role-driven, not a per-inning rate;
    # handled separately in project_pitcher_rate.
    total_pts = sum(
        t["inningsPitched"] * scoring_rules.PITCH_PTS["inningsPitched"]
        + t["earnedRuns"] * scoring_rules.PITCH_PTS["earnedRuns"]
        + t["strikeOuts"] * scoring_rules.PITCH_PTS["strikeOuts"]
        for t in totals.values()
    )
    return total_pts / total_ip if total_ip else 0.0


def apply_statcast_adjustment(rate, actual_woba, expected_xwoba):
    """Nudge a rate estimate toward what the player's underlying quality of
    contact says they *should* be doing, capped to avoid overcorrecting off
    a single-season Statcast sample."""
    if not actual_woba or not expected_xwoba:
        return rate
    ratio = expected_xwoba / actual_woba
    ratio = min(max(ratio, 1 - STATCAST_ADJUSTMENT_CAP), 1 + STATCAST_ADJUSTMENT_CAP)
    return rate * ratio


def milb_informed_prior_hitter(player_id, season, league_pts_per_pa):
    """Build a points-per-PA prior from this player's MiLB record (weighted
    across levels if he was promoted mid-season, translated per level, and
    itself shrunk toward flat league average based on level-confidence-
    weighted sample size). Falls back cleanly to the flat league average if
    the player has no MiLB record this season (e.g. an injured veteran on a
    rehab-adjacent thin MLB sample rather than an actual prospect call-up)."""
    by_level = mlb_data.get_player_milb_season_stats_by_level(player_id, season, "hitting")
    if not by_level:
        return league_pts_per_pa, {"milbPA": 0, "milbLevels": []}

    weighted_points = 0.0
    weighted_pa = 0.0
    effective_sample = 0.0
    levels_used = []
    for level_name, stat in by_level.items():
        pa = int(stat.get("plateAppearances", 0))
        if pa == 0:
            continue
        ab = int(stat.get("atBats", 0))
        h = int(stat.get("hits", 0))
        raw_points = scoring_rules.batting_points_from_totals(
            ab=ab, h=h, doubles=int(stat.get("doubles", 0)), triples=int(stat.get("triples", 0)),
            hr=int(stat.get("homeRuns", 0)), rbi=int(stat.get("rbi", 0)),
            sb=int(stat.get("stolenBases", 0)), bb=int(stat.get("baseOnBalls", 0)),
            hbp=int(stat.get("hitByPitch", 0)), runs=int(stat.get("runs", 0)),
        )
        level_rate = raw_points / pa
        translated_rate = level_rate * LEVEL_TRANSLATION_FACTOR[level_name]
        weighted_points += translated_rate * pa
        weighted_pa += pa
        effective_sample += pa * LEVEL_CONFIDENCE[level_name]
        levels_used.append({"level": level_name, "PA": pa, "rawRatePerPA": round(level_rate, 4)})

    if weighted_pa == 0:
        return league_pts_per_pa, {"milbPA": 0, "milbLevels": []}

    translated_blended_rate = weighted_points / weighted_pa
    prior = shrink(translated_blended_rate, effective_sample, league_pts_per_pa, MILB_PRIOR_K_PA)
    return prior, {"milbPA": int(weighted_pa), "milbLevels": levels_used,
                    "milbTranslatedRate": round(translated_blended_rate, 4)}


def milb_informed_prior_pitcher(player_id, season, league_pts_per_inning):
    """Same idea as milb_informed_prior_hitter, for the IP/ER/K rate component."""
    by_level = mlb_data.get_player_milb_season_stats_by_level(player_id, season, "pitching")
    if not by_level:
        return league_pts_per_inning, {"milbIP": 0, "milbLevels": []}

    weighted_points = 0.0
    weighted_ip = 0.0
    effective_sample = 0.0
    levels_used = []
    for level_name, stat in by_level.items():
        ip = mlb_data.ip_to_decimal_innings(stat.get("inningsPitched", "0.0"))
        if ip == 0:
            continue
        raw_points = (
            ip * scoring_rules.PITCH_PTS["inningsPitched"]
            + int(stat.get("earnedRuns", 0)) * scoring_rules.PITCH_PTS["earnedRuns"]
            + int(stat.get("strikeOuts", 0)) * scoring_rules.PITCH_PTS["strikeOuts"]
        )
        level_rate = raw_points / ip
        # Note: translation factor discounts *hitter* production; a pitcher's
        # equivalent adjustment runs the other way (MLB hitters are tougher
        # than MiLB hitters, so a MiLB pitcher's rate should come DOWN when
        # translated up — same direction, same factor table applied to the
        # rate itself works fine here since it's a single composite points
        # rate rather than separate offense/defense translations).
        translated_rate = level_rate * LEVEL_TRANSLATION_FACTOR[level_name]
        weighted_points += translated_rate * ip
        weighted_ip += ip
        effective_sample += ip * LEVEL_CONFIDENCE[level_name]
        levels_used.append({"level": level_name, "IP": round(ip, 1), "rawRatePerIP": round(level_rate, 4)})

    if weighted_ip == 0:
        return league_pts_per_inning, {"milbIP": 0, "milbLevels": []}

    translated_blended_rate = weighted_points / weighted_ip
    prior = shrink(translated_blended_rate, effective_sample, league_pts_per_inning, MILB_PRIOR_K_IP)
    return prior, {"milbIP": round(weighted_ip, 1), "milbLevels": levels_used,
                    "milbTranslatedRate": round(translated_blended_rate, 4)}


def project_hitter_rate(player_id, season, league_pts_per_pa, savant_row=None):
    """Returns projected fantasy points per PA for a hitter, regressed
    toward league average (or a MiLB-informed prior for thin-MLB-sample
    call-ups) and optionally Statcast-adjusted."""
    stat = mlb_data.get_player_season_stats(player_id, season, "hitting")
    pa = int(stat.get("plateAppearances", 0)) if stat else 0

    prior = league_pts_per_pa
    milb_info = {"milbPA": 0, "milbLevels": []}
    if pa < MLB_PA_THRESHOLD:
        prior, milb_info = milb_informed_prior_hitter(player_id, season, league_pts_per_pa)

    if not stat:
        return {"pointsPerPA": round(prior, 4), "PA": 0, "note": "no MLB stats — MiLB-informed" if milb_info["milbPA"] else "no stats found", **milb_info}

    ab = int(stat.get("atBats", 0))
    h = int(stat.get("hits", 0))
    raw_points = scoring_rules.batting_points_from_totals(
        ab=ab, h=h, doubles=int(stat.get("doubles", 0)), triples=int(stat.get("triples", 0)),
        hr=int(stat.get("homeRuns", 0)), rbi=int(stat.get("rbi", 0)),
        sb=int(stat.get("stolenBases", 0)), bb=int(stat.get("baseOnBalls", 0)),
        hbp=int(stat.get("hitByPitch", 0)), runs=int(stat.get("runs", 0)),
    )
    observed_rate = raw_points / pa if pa else prior
    shrunk = shrink(observed_rate, pa, prior, HITTER_SHRINKAGE_K_PA)

    if savant_row is not None:
        shrunk = apply_statcast_adjustment(
            shrunk, savant_row.get("woba"), savant_row.get("est_woba")
        )

    return {"pointsPerPA": round(shrunk, 4), "PA": pa, "rawObservedRate": round(observed_rate, 4), **milb_info}


def project_pitcher_rate(player_id, team_id, season, league_pts_per_inning, savant_row=None, team_pitching_totals=None):
    """Returns projected fantasy points per inning (IP/ER/K component only)
    plus win-share and save-share ratios used by the EV engine to project
    wins/saves from a team's projected win total in a round. Falls back to
    a MiLB-informed prior for pitchers with a thin MLB innings sample.

    Pass `team_pitching_totals` (from mlb_data.get_team_pitching_totals(season),
    fetched ONCE per run) to avoid re-pulling all 30 teams' stats on every
    single pitcher — that was happening implicitly before and was a real
    performance bug, not just a style nit."""
    stat = mlb_data.get_player_season_stats(player_id, season, "pitching")
    ip = mlb_data.ip_to_decimal_innings(stat.get("inningsPitched", "0.0")) if stat else 0.0

    prior = league_pts_per_inning
    milb_info = {"milbIP": 0, "milbLevels": []}
    if ip < MLB_IP_THRESHOLD:
        prior, milb_info = milb_informed_prior_pitcher(player_id, season, league_pts_per_inning)

    if not stat:
        return {"pointsPerIP": round(prior, 4), "IP": 0, "winsPerStart": 0, "savesPerTeamWin": 0,
                "note": "no MLB stats — MiLB-informed" if milb_info["milbIP"] else "no stats found",
                **milb_info}

    raw_points = (
        ip * scoring_rules.PITCH_PTS["inningsPitched"]
        + int(stat.get("earnedRuns", 0)) * scoring_rules.PITCH_PTS["earnedRuns"]
        + int(stat.get("strikeOuts", 0)) * scoring_rules.PITCH_PTS["strikeOuts"]
    )
    observed_rate = raw_points / ip if ip else prior
    shrunk = shrink(observed_rate, ip, prior, PITCHER_SHRINKAGE_K_IP)

    if savant_row is not None:
        shrunk = apply_statcast_adjustment(
            shrunk, savant_row.get("woba_against"), savant_row.get("est_woba_against")
        )

    games_started = int(stat.get("gamesStarted", 0))
    wins = int(stat.get("wins", 0))
    saves = int(stat.get("saves", 0))

    # Wins-per-start regressed toward team win% — see module docstring for rationale.
    if team_pitching_totals is None:
        team_pitching_totals = mlb_data.get_team_pitching_totals(season)
    team_pitching = team_pitching_totals.get(team_id, {})
    team_starts = team_pitching.get("gamesStarted", 1) or 1
    team_win_rate_prior = team_pitching.get("wins", 0) / team_starts if team_starts else 0.5
    observed_win_rate = wins / games_started if games_started else team_win_rate_prior
    wins_per_start = shrink(observed_win_rate, games_started, team_win_rate_prior, k=8)

    # Save share: this pitcher's share of the team's total wins that became
    # his saves — captures both "is he the closer" and "how often do save
    # situations arise" in one empirically-grounded number. A just-called-up
    # rookie closer prospect won't have this yet — projects to 0 until he
    # accumulates a real MLB save share, which is the honest answer (we
    # can't responsibly project MiLB saves onto an MLB closer role).
    team_wins_season = team_pitching.get("wins", 1) or 1
    saves_per_team_win = saves / team_wins_season

    return {
        "pointsPerIP": round(shrunk, 4),
        "IP": round(ip, 1),
        "gamesStarted": games_started,
        "winsPerStart": round(wins_per_start, 4),
        "savesPerTeamWin": round(saves_per_team_win, 4),
        **milb_info,
    }


def classify_hitter_role(player_id, team_id, season, last_n_days=30, splits=None, team_games=None):
    """Pass `team_games` (from mlb_data.get_team_recent_game_count, fetched
    ONCE per team) to avoid re-pulling the same team's schedule for every
    hitter on the roster — same class of bug as team_pitching_totals above."""
    gamelog = mlb_data.get_player_recent_gamelog(player_id, season, "hitting", last_n_days, splits=splits)
    if team_games is None:
        team_games = mlb_data.get_team_recent_game_count(team_id, season, last_n_days)
    games_played = len(gamelog)
    share = games_played / team_games if team_games else 0.0

    if share >= 0.70:
        role = "everyday"
    elif share >= 0.35:
        role = "platoon"
    else:
        role = "bench"

    return {"role": role, "gamesPlayed": games_played, "teamGames": team_games, "shareOfGames": round(share, 3)}


def classify_pitcher_role(player_id, team_id, season, last_n_days=30, splits=None):
    gamelog = mlb_data.get_player_recent_gamelog(player_id, season, "pitching", last_n_days, splits=splits)
    starts = sum(1 for g in gamelog if int(g["stat"].get("gamesStarted", 0)) > 0)
    saves_recent = sum(int(g["stat"].get("saves", 0)) for g in gamelog)
    appearances = len(gamelog)

    if starts > 0:
        role = "SP"
    elif saves_recent >= 3:
        role = "RP-closer"
    elif appearances > 0:
        role = "RP"
    else:
        role = "inactive/unclear"

    return {"role": role, "recentStarts": starts, "recentAppearances": appearances, "recentSaves": saves_recent}


# --- Recent form / momentum -------------------------------------------------
# Generalizes the "trailing window" idea beyond just closers: for every
# hitter and pitcher, compare their rate over the last N games actually
# played (not a calendar window — see get_player_last_n_games) against
# their season-long shrunk projection. The difference is real, computed
# momentum — this is what should drive trend arrows/charts, replacing any
# placeholder/synthetic trend data.

RECENT_FORM_GAMES = 15
RECENT_SAVE_MIN_APPEARANCES = 5  # below this, trust the season-long save share instead


def compute_recent_form_hitter(player_id, season, n_games=RECENT_FORM_GAMES, splits=None):
    games = mlb_data.get_player_last_n_games(player_id, season, "hitting", n_games, splits=splits)
    if not games:
        return None
    totals = {"atBats": 0, "hits": 0, "doubles": 0, "triples": 0, "homeRuns": 0, "rbi": 0,
              "stolenBases": 0, "baseOnBalls": 0, "hitByPitch": 0, "runs": 0, "plateAppearances": 0}
    for g in games:
        s = g.get("stat", {})
        for k in totals:
            totals[k] += int(s.get(k, 0) or 0)

    pa = totals["plateAppearances"]
    if pa == 0:
        return None
    pts = scoring_rules.batting_points_from_totals(
        ab=totals["atBats"], h=totals["hits"], doubles=totals["doubles"], triples=totals["triples"],
        hr=totals["homeRuns"], rbi=totals["rbi"], sb=totals["stolenBases"], bb=totals["baseOnBalls"],
        hbp=totals["hitByPitch"], runs=totals["runs"],
    )
    return {"recentPointsPerPA": round(pts / pa, 4), "recentPA": pa, "recentGamesUsed": len(games)}


def compute_recent_form_pitcher(player_id, season, n_games=RECENT_FORM_GAMES, splits=None):
    games = mlb_data.get_player_last_n_games(player_id, season, "pitching", n_games, splits=splits)
    if not games:
        return None
    total_ip = total_er = total_k = total_saves = relief_appearances = 0
    for g in games:
        s = g.get("stat", {})
        ip = mlb_data.ip_to_decimal_innings(s.get("inningsPitched", "0.0"))
        total_ip += ip
        total_er += int(s.get("earnedRuns", 0) or 0)
        total_k += int(s.get("strikeOuts", 0) or 0)
        total_saves += int(s.get("saves", 0) or 0)
        if int(s.get("gamesStarted", 0) or 0) == 0:
            relief_appearances += 1

    if total_ip == 0:
        return None
    pts = (
        total_ip * scoring_rules.PITCH_PTS["inningsPitched"]
        + total_er * scoring_rules.PITCH_PTS["earnedRuns"]
        + total_k * scoring_rules.PITCH_PTS["strikeOuts"]
    )
    return {
        "recentPointsPerIP": round(pts / total_ip, 4),
        "recentIP": round(total_ip, 1),
        "recentGamesUsed": len(games),
        "recentSaves": total_saves,
        "recentReliefAppearances": relief_appearances,
        "recentSaveConversionRate": round(total_saves / relief_appearances, 4) if relief_appearances else 0.0,
    }


def compute_momentum(recent_rate, projected_rate):
    """Positive = performing above projection lately (hot), negative = below
    (cold). None if there isn't enough recent data to say anything."""
    if recent_rate is None or projected_rate in (None, 0):
        return None
    return round(recent_rate - projected_rate, 4)


def pick_save_signal(recent_form_pitcher, season_saves_per_team_win):
    """Prefers the trailing-window save-conversion rate when there's enough
    recent relief sample to trust it (reacts to bullpen shuffles in real
    time); falls back to the season-long share otherwise. Always returns
    both numbers plus which one is 'primary' and why, so nothing is hidden."""
    use_recent = (
        recent_form_pitcher is not None
        and recent_form_pitcher.get("recentReliefAppearances", 0) >= RECENT_SAVE_MIN_APPEARANCES
    )
    return {
        "saveSignal": recent_form_pitcher["recentSaveConversionRate"] if use_recent else season_saves_per_team_win,
        "saveSignalSource": (
            f"trailing {recent_form_pitcher['recentGamesUsed']} games" if use_recent
            else "season-long (insufficient recent relief sample)"
        ),
        "recentSaveConversionRate": recent_form_pitcher["recentSaveConversionRate"] if recent_form_pitcher else None,
        "seasonSavesPerTeamWin": season_saves_per_team_win,
    }
