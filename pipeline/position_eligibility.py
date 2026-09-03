"""
Position eligibility, per NFBC's own rule — NOT a naive "primary position"
pull from a roster snapshot.

IMPORTANT CAVEAT, read before trusting this for a real draft:
NFBC does not appear to expose its official position-eligibility list via
any public API — it's shown on their site behind an owner login. What IS
public is the rule they've used consistently for years (documented across
their own forum posts from roughly 2008 through 2023):

    a) 20+ games at a position in the PRIOR season carries eligibility
       forward for the whole current season.
    b) 10+ games at a NEW position during the CURRENT season adds
       eligibility at that position once the 10th game is reached.
    c) If neither threshold is met at any position, the player is eligible
       only at whichever position he played most (falling back to minor
       league games if he has no MLB games at all).
    d) A player who only DH'd / pinch-hit may be UT-only.

This module REPLICATES that rule mechanically using MLB Stats API fielding
games-by-position data (both this season and last). In most cases that
should match NFBC's actual list, since their rule is itself mechanical. It
will NOT catch:
  - Whatever numeric threshold NFBC has published for THIS specific year
    (they've adjusted it before — e.g. lowered to 7 games in the shortened
    2021 season). Verify the current year's actual threshold before a real
    draft; the constants below are set to the long-standing 20/10 default.
  - Manual commissioner tie-break rulings for edge cases (rare, but they do
    happen — see their own forum posts adjudicating specific players).

Treat this as "very likely correct, not a guaranteed match to NFBC's own
list." For anything roster-critical, cross-check against NFBC's site.
"""

from . import mlb_data

PRIOR_SEASON_GAMES_THRESHOLD = 20
CURRENT_SEASON_GAMES_THRESHOLD = 10

# NFBC roster slots only have a generic OF and UT — no LF/CF/RF distinction,
# and pitchers are handled as SP/RP via role classification, not here.
FIELD_POSITIONS = ["C", "1B", "2B", "3B", "SS", "OF"]


def compute_eligible_positions(player_id, current_season, roster_listed_position=None):
    """Returns (eligible_positions: list[str], detail: dict) for a hitter.
    roster_listed_position is used only as a last-resort fallback if the
    player has no usable fielding game data at all (e.g. a true rookie with
    everything in the minors — falls back to whatever the roster snapshot
    lists rather than leaving eligibility empty)."""
    prior = mlb_data.get_player_fielding_games_by_position(player_id, current_season - 1)
    current = mlb_data.get_player_fielding_games_by_position(player_id, current_season)

    eligible = set()
    for pos, g in prior.items():
        if pos in FIELD_POSITIONS and g >= PRIOR_SEASON_GAMES_THRESHOLD:
            eligible.add(pos)
    for pos, g in current.items():
        if pos in FIELD_POSITIONS and g >= CURRENT_SEASON_GAMES_THRESHOLD:
            eligible.add(pos)

    detail = {"priorSeasonGames": prior, "currentSeasonGames": current, "fallbackUsed": False}

    if not eligible:
        # combine both seasons, take whichever field position has the most total games
        combined = {}
        for pos, g in prior.items():
            if pos in FIELD_POSITIONS:
                combined[pos] = combined.get(pos, 0) + g
        for pos, g in current.items():
            if pos in FIELD_POSITIONS:
                combined[pos] = combined.get(pos, 0) + g

        if combined:
            best = max(combined, key=combined.get)
            eligible.add(best)
            detail["fallbackUsed"] = True
        elif roster_listed_position in FIELD_POSITIONS:
            eligible.add(roster_listed_position)
            detail["fallbackUsed"] = True
        else:
            # DH-only / pinch-hit-only track — UT eligibility only, per rule (d)
            eligible.add("UT")
            detail["fallbackUsed"] = True

    positions = sorted(eligible, key=lambda p: FIELD_POSITIONS.index(p) if p in FIELD_POSITIONS else 99)
    return positions, detail
