# NFBC Postseason Holdem — Automation

Two independent pipelines, both scheduled via GitHub Actions and writing to
the same Firestore project:

1. **Results scorer** (`scripts/nfbc_holdem_scoring.py`) — what already
   happened. Pulls completed box scores and scores them per NFBC Postseason
   Holdem rules. Only useful once the postseason has started.
2. **Projection pipeline** (`pipeline/`) — what's likely to happen. Team
   power ratings + market odds + player rate/role projections, usable
   *right now* during the regular season and automatically sharpening into
   a full round-by-round win-probability + player EV model as the bracket
   becomes real. See "How the projection model works" below.

## What's in here

```
scripts/nfbc_holdem_scoring.py     # results scorer (CLI + Firestore push)
pipeline/
  mlb_data.py                      # shared MLB Stats API client
  odds_client.py                   # The Odds API client (sportsbook lines)
  savant_data.py                   # Baseball Savant expected-stats pull
  scoring_rules.py                 # single source of truth for point values
  team_model.py                    # Pythagorean/log5/Monte Carlo series sim
  player_model.py                  # rate shrinkage + role classification
  ev_engine.py                     # orchestration — run this one
run_projections.py                 # convenience entrypoint for the pipeline
requirements.txt
.github/workflows/daily-pull.yml           # results scorer schedule
.github/workflows/projection-update.yml    # projection pipeline schedule
```

## One-time setup

### 1. Push this to a GitHub repo
```bash
git init
git add .
git commit -m "NFBC Holdem daily stat automation"
git remote add origin <your-repo-url>
git push -u origin main
```

### 2. Create a Firestore database (if you haven't already)
In the Firebase console: **Build → Firestore Database → Create database**.
Start in production mode — the service account below bypasses security rules
anyway, so rules just need to lock out public read/write (default is fine).

### 3. Create a service account key
Firebase console → **Project settings → Service accounts → Generate new
private key**. This downloads a JSON file — treat it like a password.

### 4. Add secrets to GitHub
Repo → **Settings → Secrets and variables → Actions → New repository secret**
- `FIREBASE_SERVICE_ACCOUNT_JSON` — paste the **entire contents** of the downloaded JSON file
- `ODDS_API_KEY` — free key from [the-odds-api.com](https://the-odds-api.com) (500 credits/month free
  tier, no card required). Powers the market-odds side of the projection
  pipeline; the results scorer doesn't need it.

**Never commit either value into a file in this repo** — both scripts read
them from environment variables (`FIREBASE_SERVICE_ACCOUNT_JSON`,
`ODDS_API_KEY`) so they only ever exist as GitHub secrets / your local shell
environment.

### 5. Test it manually before relying on the schedule
Repo → **Actions → NFBC Holdem Daily Stat Pull → Run workflow**, and enter a
date from a *past* postseason (e.g. `2025-10-15`) so there's real data to
pull. Check:
- The Action run succeeds (green check)
- Firestore has a new `player_game_stats` collection with documents
- A `pull_runs` collection entry was logged
- The CSV is attached as a downloadable workflow artifact

## How the schedule works

The workflow runs twice daily (5am and 11am ET) and pulls **yesterday's**
completed games by default — playoff games often end late, so pulling the
morning after is more reliable than trying to catch same-day completion.
Only games with status `Final` are pulled by default (no partial stat lines
from in-progress games).

You can also trigger it manually anytime via **Actions → Run workflow** and
specify any date, which is useful for backfilling or re-running a day if a
game's stats were corrected after an official scoring review (safe — writes
are idempotent per `date_gamePk_playerId`, so re-running just updates existing
docs).

## Important: GitHub auto-disables idle scheduled workflows

GitHub automatically disables scheduled (`cron`) workflows after **60 days
with no repository activity**. Since the season doesn't start until October,
if you set this up now and don't touch the repo again, it may need to be
manually re-enabled in the Actions tab right before the postseason starts.
Worth a calendar reminder in early October.

## How the projection model works

Run it directly:
```bash
python run_projections.py --season 2026                          # print only
python run_projections.py --season 2026 --push-firestore          # + write to Firestore
python run_projections.py --season 2026 --push-firestore --contenders-only  # faster, top-half teams only
```

### The core idea

Season-to-date stats are a weak predictor of postseason fantasy points in
this format, for two reasons specific to NFBC Postseason Holdem:

1. **The round multiplier dominates.** A merely-good player on a team that
   reaches the World Series (3x that round) outscores a star on a team
   swept in the Division Series. The highest-leverage prediction isn't
   "who's been good" — it's **which teams advance**.
2. **This is counting-stat scoring with almost no out penalty (-0.25).**
   Playing time and role certainty matter more than rate efficiency.

So the model has two independent halves that get multiplied together:
**team survival probability** x **player rate-per-opportunity x expected
opportunities**.

### Team survival probability (`pipeline/team_model.py`)

- **Pre-bracket** (now, through the Wild Card round): no real matchups
  exist yet, so there's nothing honest to simulate round-by-round. The
  pipeline instead stores each team's **Pythagorean win%** (runs
  scored/allowed based true-talent estimate, more predictive than raw W-L)
  and, when available, the **market-implied World Series probability**
  (de-vigged sportsbook futures odds) — both useful research signals on
  their own.
- **Once a round's real matchup is scheduled** (MLB Stats API's postseason
  schedule endpoint reflects this the moment it's official): the model
  computes each team's per-game win probability via the **log5 method**
  on Pythagorean win%, blends it with real de-vigged sportsbook series
  odds (logit-averaged, 50/50 by default), then **Monte Carlo simulates
  the series** (10,000 iterations) to get a series-win probability.
- The pipeline never guesses a hypothetical future bracket — it only ever
  projects series that MLB has actually scheduled. That's a deliberate
  simplification: Round 2/3 opponents aren't real until earlier rounds
  finish, so trying to model them earlier would mean modeling noise. Each
  round's projection gets meaningfully better the moment that round's
  matchup is set — which is also exactly when you need it, since Holdem's
  lock times are tied to each round's first game.

### Player rate + role projection (`pipeline/player_model.py`)

- **Rate**: season totals converted to points-per-PA (hitters) or
  points-per-inning (pitchers), then **regressed toward the league-average
  rate** using Empirical-Bayes shrinkage weighted by sample size — a
  player with 500 PA is trusted more than one with 60. Optionally
  cross-checked against Baseball Savant's expected stats (xwOBA vs. actual
  wOBA) to catch over/underperformance not yet visible in the raw shrunk
  rate.
- **Wins/saves**: modeled separately from the IP/ER/K rate, since they're
  role-driven rather than a clean per-inning rate. Wins-per-start is
  shrunk toward the pitcher's team win%; saves are projected as this
  pitcher's share of the team's *wins* this season (`saves ÷ team wins`),
  which empirically captures both "is this player the closer" and "how
  often do save situations arise" in one number.
- **Role**: everyday/platoon/bench for hitters and SP/RP/closer for
  pitchers, classified from **trailing-30-day usage** (not season-long),
  since recent usage is the best signal for current role after trades,
  injuries, or demotions.

### Minor league call-ups (`pipeline/player_model.py` + `pipeline/mlb_data.py`)

A player with a thin MLB sample (below `MLB_PA_THRESHOLD` / `MLB_IP_THRESHOLD`)
doesn't just shrink to flat league average — the same public MLB Stats API
also serves MiLB data (same endpoints, different `sportId` per level: AAA,
AA, High-A, Single-A, Rookie), so the model pulls the player's record across
every level played this season, translates it toward MLB-quality competition
using rough historical MLE-style factors, weights lower levels far less
confidently (translation accuracy is well-documented to degrade sharply
below AAA), and shrinks that into a MiLB-informed prior *before* blending in
whatever small MLB sample exists. A hot AAA prospect and a replacement-level
scrub who both have 15 MLB PA will not get the same projection. Verified
with a mocked test: identical 15-PA MLB samples produced 0.695 pts/PA for a
prospect with a strong AAA season vs. 0.516 (near league average) for one
with no MiLB track record — see the level-translation constants at the top
of `player_model.py` if you want to tune them (they're explicitly flagged as
approximations, not empirically re-derived here).

One related honest limitation: role classification (everyday/platoon/bench,
SP/RP/closer) uses trailing-30-day *MLB* usage only. A player called up
within the last few days will show a small, uncertain sample there too —
that's real uncertainty being reflected accurately, not a bug, but worth
knowing going in.

### Known simplifications (documented, not hidden)

- No probable-starting-pitcher-level matchup adjustment within a series yet.
- No park/weather adjustment yet (cold-weather October games at
  NY/Boston/Chicago/Cleveland/Detroit/Milwaukee modestly suppress
  offense — a reasonable future addition via the free NWS API).
- Role classification is a threshold heuristic, not a full playing-time model.
- Win/save shares are season-long ratios, not further game-context-adjusted.

These are all isolated, addressable next steps sitting behind clear function
boundaries (`team_model.py`, `player_model.py`) — nothing here needs a
rewrite to improve later, just a better estimate plugged into an existing slot.

## Wiring the dashboard to live data

`ui/dashboard.html` reads directly from Firestore in the browser (no backend
needed for reads — writes are locked down to server-side only via the
security rules from the Firebase setup step). One step required:

Open `ui/dashboard.html`, find the `firebaseConfig` object near the top of
the `<script>` block, and paste in your actual values from Firebase step 4
above. Leave it as the placeholder and the dashboard runs on sample data
automatically — no code branching needed, it's a single `isFirebaseConfigured()`
check.

If Firebase is configured but a collection is empty (pipeline hasn't run
yet) or a read fails for any reason, it also falls back to sample data
rather than showing a blank page — the status pill in the header tells you
which mode you're in ("Live data from Firestore" vs. a sample-data warning).

## Data model in Firestore

**`team_power_ratings`** — one doc per team per season
- doc ID: `{season}_{teamId}`
- fields: teamName, division, league, runsScored, runsAllowed,
  pythagoreanWinPct, marketWSOdds

**`team_round_probabilities`** — one doc per team currently in a real series
- doc ID: `{season}_{teamId}`
- fields: currentRound, advanceCurrentRoundProb, modelGameProb, marketGameProb

**`player_projections`** — one doc per rostered player
- doc ID: `{season}_{playerId}`
- fields: team, position, pointsPerPA or pointsPerIP, role, winsPerStart /
  savesPerTeamWin (pitchers), currentRound, advanceCurrentRoundProb,
  roundMultiplier (once a bracket round is real), momentum (recent-vs-
  projected delta), recentPointsPerPA/IP, saveSignal + saveSignalSource
  (trailing-window vs. season-long — see "Recent form" below)

**`team_rating_history`** / **`player_projection_history`** — one doc per
team/player per calendar date (`{date}_{teamId}` / `{date}_{playerId}`).
Written alongside the main upsert-in-place docs above specifically so the
dashboard's trend charts have real history to plot instead of a single
snapshot. Accumulates daily; first day will only have one data point.

**`system_meta/last_run`** — single doc, overwritten each run: `ranAt`
(server timestamp — this is what the dashboard's "Updated" indicator
reads), `season`, `teamsProjected`, `playersProjected`, `bracketState`.

### Recent form / momentum (generalized beyond closers)

Every hitter and pitcher gets a `momentum` field: their rate over their
actual last 15 games played (not a calendar window — see
`get_player_last_n_games` in `mlb_data.py`) compared against their season-
long shrunk projection. Positive = trending hot, negative = cold. This is
real computed data, not a placeholder, and it's what should drive any
trend arrow/indicator in the UI.

For pitcher saves specifically, `pick_save_signal()` in `player_model.py`
prefers this trailing-window save-conversion rate over the season-long
share once there's enough recent relief sample (5+ appearances) — reacts
to bullpen shuffles in real time instead of waiting for a whole season's
data to catch up. Both numbers are always stored (`saveSignal` +
`saveSignalSource` tells you which one is active and why), nothing hidden.

**`player_game_stats`** — one doc per player per game
- doc ID: `{date}_{gamePk}_{playerId}`
- fields: date, gamePk, team, playerId, player, position, raw stat
  categories (AB, R, H, 1B, 2B, 3B, HR, RBI, SB, BB, HBP, Outs, IP, ER, K, W,
  SV), battingPts, pitchingPts, totalPts

**`pull_runs`** — one doc per workflow execution, for debugging/visibility
- fields: ranAt (server timestamp), rowsWritten, dates

Note: **no round multiplier is applied here** — that requires knowing your
actual drafted roster and which round each player was added, which lives
outside the stats layer. That aggregation logic is the natural next build.
