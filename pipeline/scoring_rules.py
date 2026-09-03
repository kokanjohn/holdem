"""
NFBC Postseason Holdem scoring rules — the one place these numbers live.
Both scripts/nfbc_holdem_scoring.py (actual results) and pipeline/player_model.py
(projections) import from here so the two can never drift out of sync.
"""

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
    "outs": -0.25,          # derived: AB - H
}

PITCH_PTS = {
    "inningsPitched": 1.00,  # per full (decimal) inning
    "earnedRuns": -1.00,
    "strikeOuts": 1.00,
    "wins": 4.00,
    "saves": 4.00,
}


def batting_points_from_totals(pa=0, ab=0, h=0, doubles=0, triples=0, hr=0,
                                rbi=0, sb=0, bb=0, hbp=0, runs=0):
    singles = h - doubles - triples - hr
    outs = ab - h
    return (
        runs * HIT_PTS["runs"] +
        singles * HIT_PTS["singles"] +
        doubles * HIT_PTS["doubles"] +
        triples * HIT_PTS["triples"] +
        hr * HIT_PTS["homeRuns"] +
        rbi * HIT_PTS["rbi"] +
        sb * HIT_PTS["stolenBases"] +
        bb * HIT_PTS["baseOnBalls"] +
        hbp * HIT_PTS["hitByPitch"] +
        outs * HIT_PTS["outs"]
    )


def pitching_points_from_totals(ip_decimal=0.0, er=0, k=0, wins=0, saves=0):
    return (
        ip_decimal * PITCH_PTS["inningsPitched"] +
        er * PITCH_PTS["earnedRuns"] +
        k * PITCH_PTS["strikeOuts"] +
        wins * PITCH_PTS["wins"] +
        saves * PITCH_PTS["saves"]
    )
