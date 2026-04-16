# playoff-pool
NHL Playoff Pool Draft Model — 2026

## Overview

A Python-based draft model for a 14-team snake-draft NHL playoff pool. The model
ranks all ~200 key playoff skaters by **Draft Score** so you can maximise total
expected fantasy points across the entire 2026 playoffs.

### Scoring system

| Event | Points |
|---|---|
| Goal | 1.5 |
| Assist | 1.0 |
| Power-play point (bonus) | +0.5 |
| OT goal (bonus) | +1.0 |
| OT assist (bonus) | +0.5 |
| **Team** Win | 1.5 |
| **Team** Shutout | 1.5 |

### League format

- 14 fantasy teams · snake draft · 11 roster spots per team  
- Each participant selects one NHL team (stacking allowed)

## How the model works

1. **PPG\_model** — per-game fantasy point projection  
   `ppg = gpg×1.5 + apg×1.0 + ppp_pg×0.5 + (gpg×1.0 + apg×0.5)×0.10`  
   Per-game rates are weighted blends of regular-season (60 %) and career playoff
   (40 %) data.

2. **EGP** (Expected Games Played) — derived from each team's round-by-round
   series-win probability using an exact best-of-7 binomial model.  
   `EGP = Σ_r P(reach round r) × E[games in series r]`

3. **Draft Score**  
   `Draft Score = PPG_model × EGP × (1 + team_strength_boost)`  
   The team-strength boost scales with the team's implied Stanley Cup probability
   relative to the 1/16 league average.

## 2026 Playoff teams

Eastern: FLA · TBL · OTT · CAR · BOS · NYR · WSH · NJD  
Western: EDM · COL · VGK · DAL · VAN · WPG · LAK · NSH

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Full report to stdout
python playoff_model.py

# Also export rankings.csv
python playoff_model.py --csv
```

## Output

- **EGP table** — every team's expected games per round and Cup probability  
- **Top 200 players** ranked by Draft Score (team, position, line, PP unit, scores)  
- **Best picks by draft window** — early (rounds 1-2), mid (3-6), late (7-11)  
- **Top 3 teams to stack** — combined player + Win/Shutout value  
- **High-value sleepers** — PP1 players or deep-run teams ranked outside the top 75
