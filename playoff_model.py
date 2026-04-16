#!/usr/bin/env python3
"""
NHL Playoff Pool Draft Model — 2026
====================================

Scoring System
--------------
  Goals                = 1.5 pts
  Assists              = 1.0 pts
  Power-play point     = +0.5 bonus (stacks on top of base G/A pts)
  OT goal              = +1.0 bonus (stacks on top of G pts)
  OT assist            = +0.5 bonus (stacks on top of A pts)

  Team Win             = 1.5 pts   (for the one NHL team selected per roster)
  Team Shutout         = 1.5 pts

League Format
-------------
  14 fantasy teams · snake draft · 11 roster spots per team
  Each participant selects one NHL team (stacking allowed)

Objective: Maximise total expected fantasy points across all playoff rounds.

PPG Model
---------
  Playoff per-game rates are weighted blends:
    60 % regular-season rate  +  40 % career playoff rate
  (where career playoff data is sparse the RS rate is used at full weight)

Draft Score
-----------
  Draft Score = PPG_model × EGP × (1 + team_strength_boost)

  EGP (Expected Games Played) is derived from each team's round-by-round
  series-win probabilities using an exact best-of-7 probability model.

  team_strength_boost scales linearly from the team's implied Cup probability
  relative to the league average (1/16).

Usage
-----
  python playoff_model.py              # full report to stdout
  python playoff_model.py --csv        # also writes rankings.csv
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import Optional

import numpy as np
import pandas as pd

# ═══════════════════════════════════════════════════════════════════════════════
# SCORING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GOAL_PTS         = 1.5
ASSIST_PTS       = 1.0
PPP_BONUS        = 0.5    # extra pts per power-play point (goal or assist)
OT_GOAL_BONUS    = 1.0    # extra pts when the goal is an OT game-winner
OT_ASSIST_BONUS  = 0.5    # extra pts when the assist is on an OT game-winner
TEAM_WIN_PTS     = 1.5
TEAM_SHUTOUT_PTS = 1.5

# Scales how strongly a team's Cup probability above/below league average
# translates into a player draft-score multiplier.  The resulting boost is
# clamped to ±TEAM_BOOST_CAP.
TEAM_BOOST_FACTOR = 0.15
TEAM_BOOST_CAP    = 0.25

# ═══════════════════════════════════════════════════════════════════════════════
# LEAGUE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

NUM_FANTASY_TEAMS = 14
ROSTER_SIZE       = 11
TOTAL_PICKS       = NUM_FANTASY_TEAMS * ROSTER_SIZE   # 154

# Draft-window pick boundaries (snake, 14 teams × 11 rounds)
EARLY_PICKS = range(1, 29)          # rounds 1-2   (picks  1-28)
MID_PICKS   = range(29, 85)         # rounds 3-6   (picks 29-84)
LATE_PICKS  = range(85, 155)        # rounds 7-11  (picks 85-154)

# Fraction of all playoff points that carry an OT bonus
# (~25 % of games reach OT; ~10 % of a player's points are OT points)
OT_RATE = 0.10

# ═══════════════════════════════════════════════════════════════════════════════
# 2026 NHL PLAYOFF TEAMS
# ═══════════════════════════════════════════════════════════════════════════════
#
# cup_prob  – implied from betting odds, normalised so total = 1.00
# r{n}_win  – P(win that series | the team has reached that round)
# gpg_for   – goals per game scored (team offence strength indicator)
# so_rate   – fraction of games the team's goalie posts a shutout
#
TEAMS: dict[str, dict] = {
    # ── EASTERN CONFERENCE ──────────────────────────────────────────────────────
    # R1 matchups (confirmed): CAR vs PIT(WC2), TBL vs MTL, BUF vs PHI,
    #                          NYI vs WSH(WC1)
    # cup_prob values are implied-odds estimates normalised to sum = 1.00.
    "CAR": dict(
        name="Carolina Hurricanes",   conf="E", seed=1,
        cup_prob=0.09,
        r1_win=0.65, r2_win=0.58, r3_win=0.52, r4_win=0.48,
        gpg_for=3.10, so_rate=0.10,
    ),
    "TBL": dict(
        name="Tampa Bay Lightning",   conf="E", seed=2,
        cup_prob=0.09,
        r1_win=0.62, r2_win=0.57, r3_win=0.52, r4_win=0.48,
        gpg_for=3.25, so_rate=0.08,
    ),
    "BUF": dict(
        name="Buffalo Sabres",        conf="E", seed=3,
        cup_prob=0.06,
        r1_win=0.55, r2_win=0.50, r3_win=0.46, r4_win=0.42,
        gpg_for=3.20, so_rate=0.06,
    ),
    "MTL": dict(
        name="Montreal Canadiens",    conf="E", seed=4,
        cup_prob=0.04,
        r1_win=0.40, r2_win=0.46, r3_win=0.42, r4_win=0.38,
        gpg_for=2.95, so_rate=0.07,
    ),
    "NYI": dict(
        name="New York Islanders",    conf="E", seed=5,
        cup_prob=0.03,
        r1_win=0.52, r2_win=0.48, r3_win=0.44, r4_win=0.40,
        gpg_for=2.85, so_rate=0.09,
    ),
    "PHI": dict(
        name="Philadelphia Flyers",   conf="E", seed=6,
        cup_prob=0.05,
        r1_win=0.47, r2_win=0.48, r3_win=0.44, r4_win=0.40,
        gpg_for=2.95, so_rate=0.07,
    ),
    "WSH": dict(
        name="Washington Capitals",   conf="E", seed=7,
        cup_prob=0.04,
        r1_win=0.50, r2_win=0.47, r3_win=0.43, r4_win=0.39,
        gpg_for=3.05, so_rate=0.07,
    ),
    "PIT": dict(
        name="Pittsburgh Penguins",   conf="E", seed=8,
        cup_prob=0.02,
        r1_win=0.37, r2_win=0.42, r3_win=0.38, r4_win=0.35,
        gpg_for=2.90, so_rate=0.07,
    ),
    # ── WESTERN CONFERENCE ──────────────────────────────────────────────────────
    # R1 matchups (confirmed): COL vs ANA(WC2), DAL vs MIN, VGK vs UTA,
    #                          EDM vs LAK
    "COL": dict(
        name="Colorado Avalanche",    conf="W", seed=1,
        cup_prob=0.17,
        r1_win=0.72, r2_win=0.64, r3_win=0.57, r4_win=0.53,
        gpg_for=3.50, so_rate=0.05,
    ),
    "DAL": dict(
        name="Dallas Stars",          conf="W", seed=2,
        cup_prob=0.09,
        r1_win=0.60, r2_win=0.55, r3_win=0.50, r4_win=0.46,
        gpg_for=3.05, so_rate=0.09,
    ),
    "MIN": dict(
        name="Minnesota Wild",        conf="W", seed=3,
        cup_prob=0.05,
        r1_win=0.42, r2_win=0.47, r3_win=0.43, r4_win=0.39,
        gpg_for=3.10, so_rate=0.07,
    ),
    "VGK": dict(
        name="Vegas Golden Knights",  conf="W", seed=4,
        cup_prob=0.08,
        r1_win=0.58, r2_win=0.53, r3_win=0.49, r4_win=0.45,
        gpg_for=3.05, so_rate=0.09,
    ),
    "UTA": dict(
        name="Utah Mammoth",          conf="W", seed=5,
        cup_prob=0.04,
        r1_win=0.44, r2_win=0.45, r3_win=0.41, r4_win=0.37,
        gpg_for=3.05, so_rate=0.08,
    ),
    "EDM": dict(
        name="Edmonton Oilers",       conf="W", seed=6,
        cup_prob=0.11,
        r1_win=0.62, r2_win=0.57, r3_win=0.52, r4_win=0.48,
        gpg_for=3.45, so_rate=0.06,
    ),
    "LAK": dict(
        name="Los Angeles Kings",     conf="W", seed=7,
        cup_prob=0.03,
        r1_win=0.40, r2_win=0.43, r3_win=0.39, r4_win=0.36,
        gpg_for=2.82, so_rate=0.10,
    ),
    "ANA": dict(
        name="Anaheim Ducks",         conf="W", seed=8,
        cup_prob=0.01,
        r1_win=0.30, r2_win=0.38, r3_win=0.35, r4_win=0.32,
        gpg_for=2.85, so_rate=0.07,
    ),
}

# ═══════════════════════════════════════════════════════════════════════════════
# EXPECTED GAMES PLAYED (EGP) — MATHEMATICAL ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def _series_win_prob(p_game: float) -> float:
    """P(team wins best-of-7) given per-game win probability *p_game*."""
    return sum(
        math.comb(k - 1, 3) * p_game ** 4 * (1 - p_game) ** (k - 4)
        for k in range(4, 8)
    )


def _pgame_from_series_prob(p_series: float) -> float:
    """Binary-search for per-game win probability that yields *p_series*."""
    lo, hi = 0.01, 0.99
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if _series_win_prob(mid) < p_series:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _expected_series_games(p_game: float) -> float:
    """Expected number of games played in a best-of-7 series."""
    p, q = p_game, 1.0 - p_game
    return sum(
        k * math.comb(k - 1, 3) * (p ** 4 * q ** (k - 4) + q ** 4 * p ** (k - 4))
        for k in range(4, 8)
    )


def compute_team_egp(team_code: str) -> dict:
    """
    Return Expected Games Played (EGP) and round-level breakdown for *team_code*.

    EGP = Σ_r  P(reach round r) × E[games played in round r]

    A team plays all games in a series regardless of whether it wins or loses,
    so EGP compounds as the team advances.
    """
    t = TEAMS[team_code]
    r_probs = [t["r1_win"], t["r2_win"], t["r3_win"], t["r4_win"]]

    # Probability of reaching each round
    p_reach = [1.0]
    for rw in r_probs[:-1]:
        p_reach.append(p_reach[-1] * rw)

    egp_by_round: list[float] = []
    for rnd, p_series in enumerate(r_probs):
        p_game = _pgame_from_series_prob(p_series)
        eg     = _expected_series_games(p_game)
        egp_by_round.append(p_reach[rnd] * eg)

    return {
        "egp_total":  sum(egp_by_round),
        "egp_r1":     egp_by_round[0],
        "egp_r2":     egp_by_round[1],
        "egp_r3":     egp_by_round[2],
        "egp_r4":     egp_by_round[3],
        "p_reach_r2": p_reach[1],
        "p_reach_r3": p_reach[2],
        "p_reach_r4": p_reach[3],
    }


# Pre-compute EGP for every team once
TEAM_EGP: dict[str, dict] = {t: compute_team_egp(t) for t in TEAMS}

# ═══════════════════════════════════════════════════════════════════════════════
# PLAYER DATABASE
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each entry: (name, team, pos, line, pp_unit, gpg, apg, ppp_pg)
#
#   pos     – C / LW / RW / D
#   line    – 1 / 2 / 3 / 4  (0 for D pairs)
#   pp_unit – 'PP1' / 'PP2' / '-'
#   gpg     – goals per game (playoff-adjusted, 60% RS + 40% career playoff)
#   apg     – assists per game (same weighting)
#   ppp_pg  – power-play points per game
#             (these points are already counted inside gpg/apg; the PPP_BONUS
#              is applied separately in the PPG model)
#
_RAW_PLAYERS: list[tuple] = [

    # ── EDMONTON OILERS (EDM) — W#6 seed ─────────────────────────────────────
    # Core unchanged; Jeff Skinner (signed UFA 2024) fills energy LW spot.
    ("Connor McDavid",        "EDM", "C",  1, "PP1", 0.40, 0.78, 0.55),
    ("Leon Draisaitl",        "EDM", "C",  1, "PP1", 0.38, 0.62, 0.52),
    ("Zach Hyman",            "EDM", "LW", 1, "PP1", 0.28, 0.26, 0.18),
    ("Ryan Nugent-Hopkins",   "EDM", "C",  2, "PP2", 0.18, 0.30, 0.20),
    ("Evan Bouchard",         "EDM", "D",  0, "PP1", 0.15, 0.42, 0.38),
    ("Darnell Nurse",         "EDM", "D",  0, "PP2", 0.05, 0.20, 0.12),
    ("Mattias Ekholm",        "EDM", "D",  0, "-",   0.04, 0.18, 0.06),
    ("Evander Kane",          "EDM", "LW", 2, "PP2", 0.22, 0.22, 0.15),
    ("Jeff Skinner",          "EDM", "LW", 2, "PP2", 0.16, 0.18, 0.12),
    ("Connor Brown",          "EDM", "RW", 2, "-",   0.12, 0.18, 0.05),
    ("Ryan McLeod",           "EDM", "C",  3, "-",   0.10, 0.15, 0.04),
    ("Philip Broberg",        "EDM", "D",  0, "PP2", 0.08, 0.22, 0.15),
    ("Warren Foegele",        "EDM", "LW", 3, "-",   0.10, 0.12, 0.04),
    ("Sam Carrick",           "EDM", "C",  4, "-",   0.05, 0.07, 0.02),

    # ── COLORADO AVALANCHE (COL) — W#1 seed ──────────────────────────────────
    # Rantanen traded to CAR (Jan 2025); Martin Necas arrives in return.
    ("Nathan MacKinnon",      "COL", "C",  1, "PP1", 0.42, 0.72, 0.52),
    ("Martin Necas",          "COL", "RW", 1, "PP1", 0.28, 0.42, 0.30),
    ("Cale Makar",            "COL", "D",  0, "PP1", 0.18, 0.55, 0.42),
    ("Gabriel Landeskog",     "COL", "LW", 1, "PP2", 0.22, 0.32, 0.18),
    ("Valeri Nichushkin",     "COL", "RW", 2, "PP2", 0.25, 0.28, 0.15),
    ("Devon Toews",           "COL", "D",  0, "PP1", 0.10, 0.32, 0.24),
    ("Artturi Lehkonen",      "COL", "LW", 2, "-",   0.18, 0.20, 0.08),
    ("Ross Colton",           "COL", "C",  3, "-",   0.12, 0.14, 0.06),
    ("Samuel Girard",         "COL", "D",  0, "-",   0.06, 0.18, 0.08),
    ("Evan Rodrigues",        "COL", "C",  2, "PP2", 0.14, 0.20, 0.12),
    ("Bowen Byram",           "COL", "D",  0, "PP2", 0.10, 0.24, 0.18),
    ("Miles Wood",            "COL", "LW", 3, "-",   0.08, 0.10, 0.03),
    ("Jonathan Drouin",       "COL", "LW", 2, "PP2", 0.14, 0.24, 0.16),

    # ── TAMPA BAY LIGHTNING (TBL) — E#2 seed ─────────────────────────────────
    # Stamkos to NSH (UFA 2024); Sergachev traded to UTA.
    # Jake Guentzel acquired at 2024 trade deadline.
    ("Nikita Kucherov",       "TBL", "RW", 1, "PP1", 0.35, 0.72, 0.50),
    ("Brayden Point",         "TBL", "C",  1, "PP1", 0.42, 0.48, 0.35),
    ("Victor Hedman",         "TBL", "D",  0, "PP1", 0.10, 0.45, 0.32),
    ("Jake Guentzel",         "TBL", "LW", 1, "PP1", 0.32, 0.35, 0.25),
    ("Brandon Hagel",         "TBL", "LW", 2, "PP2", 0.20, 0.28, 0.15),
    ("Anthony Cirelli",       "TBL", "C",  2, "-",   0.14, 0.20, 0.06),
    ("Nicholas Paul",         "TBL", "LW", 2, "PP2", 0.12, 0.20, 0.10),
    ("Tanner Jeannot",        "TBL", "LW", 3, "-",   0.10, 0.12, 0.04),
    ("Erik Cernak",           "TBL", "D",  0, "-",   0.04, 0.12, 0.04),
    ("Tyler Motte",           "TBL", "RW", 3, "-",   0.08, 0.08, 0.03),
    ("Luke Glendening",       "TBL", "C",  4, "-",   0.06, 0.06, 0.02),

    # ── VEGAS GOLDEN KNIGHTS (VGK) — W#4 seed ────────────────────────────────
    # Marchessault to NSH (UFA 2023); Tomas Hertl & Ivan Barbashev added.
    ("Jack Eichel",           "VGK", "C",  1, "PP1", 0.30, 0.48, 0.38),
    ("Mark Stone",            "VGK", "RW", 1, "PP1", 0.22, 0.38, 0.25),
    ("Tomas Hertl",           "VGK", "C",  2, "PP1", 0.22, 0.32, 0.20),
    ("Ivan Barbashev",        "VGK", "LW", 1, "PP2", 0.18, 0.25, 0.15),
    ("William Karlsson",      "VGK", "C",  3, "PP2", 0.18, 0.25, 0.15),
    ("Alex Pietrangelo",      "VGK", "D",  0, "PP1", 0.08, 0.28, 0.18),
    ("Shea Theodore",         "VGK", "D",  0, "PP1", 0.10, 0.30, 0.22),
    ("Chandler Stephenson",   "VGK", "C",  2, "PP2", 0.15, 0.26, 0.15),
    ("Nicolas Roy",           "VGK", "C",  3, "-",   0.08, 0.12, 0.04),
    ("Keegan Kolesar",        "VGK", "RW", 3, "-",   0.08, 0.08, 0.03),
    ("Brett Howden",          "VGK", "C",  3, "-",   0.06, 0.10, 0.04),
    ("William Carrier",       "VGK", "LW", 4, "-",   0.06, 0.06, 0.02),

    # ── CAROLINA HURRICANES (CAR) — E#1 seed ─────────────────────────────────
    # Mikko Rantanen acquired from COL (Jan 2025 trade).
    ("Sebastian Aho",         "CAR", "C",  1, "PP1", 0.28, 0.38, 0.28),
    ("Mikko Rantanen",        "CAR", "RW", 1, "PP1", 0.35, 0.55, 0.42),
    ("Andrei Svechnikov",     "CAR", "LW", 2, "PP1", 0.32, 0.38, 0.22),
    ("Seth Jarvis",           "CAR", "C",  2, "PP1", 0.26, 0.30, 0.20),
    ("Jesperi Kotkaniemi",    "CAR", "C",  3, "-",   0.14, 0.20, 0.08),
    ("Brady Skjei",           "CAR", "D",  0, "PP2", 0.08, 0.26, 0.18),
    ("Jaccob Slavin",         "CAR", "D",  0, "-",   0.04, 0.16, 0.06),
    ("Brent Burns",           "CAR", "D",  0, "PP1", 0.08, 0.24, 0.18),
    ("Jesper Fast",           "CAR", "RW", 3, "-",   0.06, 0.08, 0.03),
    ("Jordan Staal",          "CAR", "C",  3, "-",   0.06, 0.10, 0.04),
    ("Dmitry Orlov",          "CAR", "D",  0, "PP2", 0.06, 0.18, 0.10),
    ("Jordan Martinook",      "CAR", "LW", 3, "-",   0.06, 0.10, 0.03),

    # ── DALLAS STARS (DAL) — W#2 seed ────────────────────────────────────────
    # Pavelski retired; Matt Duchene signed DAL (UFA 2023).
    ("Jason Robertson",       "DAL", "LW", 1, "PP1", 0.38, 0.42, 0.32),
    ("Roope Hintz",           "DAL", "C",  1, "PP1", 0.30, 0.38, 0.25),
    ("Miro Heiskanen",        "DAL", "D",  0, "PP1", 0.08, 0.32, 0.22),
    ("Tyler Seguin",          "DAL", "C",  2, "PP2", 0.20, 0.30, 0.20),
    ("Matt Duchene",          "DAL", "C",  2, "PP1", 0.20, 0.28, 0.18),
    ("Mason Marchment",       "DAL", "LW", 2, "PP2", 0.18, 0.22, 0.14),
    ("Logan Stankoven",       "DAL", "C",  3, "PP2", 0.14, 0.18, 0.10),
    ("Jamie Benn",            "DAL", "LW", 3, "-",   0.10, 0.14, 0.05),
    ("Esa Lindell",           "DAL", "D",  0, "PP2", 0.05, 0.18, 0.10),
    ("Radek Faksa",           "DAL", "C",  4, "-",   0.06, 0.08, 0.03),
    ("Thomas Harley",         "DAL", "D",  0, "PP1", 0.08, 0.22, 0.16),

    # ── LOS ANGELES KINGS (LAK) — W#7 seed ───────────────────────────────────
    # Quinton Byfield (franchise C); Dubois & DeBrusk signed as UFAs.
    # Kopitar moves to L3; Arvidsson inactive (chronic injury).
    ("Quinton Byfield",       "LAK", "C",  1, "PP1", 0.28, 0.42, 0.28),
    ("Kevin Fiala",           "LAK", "LW", 1, "PP1", 0.22, 0.30, 0.22),
    ("Adrian Kempe",          "LAK", "LW", 2, "PP1", 0.22, 0.28, 0.20),
    ("Pierre-Luc Dubois",     "LAK", "C",  2, "PP2", 0.18, 0.28, 0.16),
    ("Jake DeBrusk",          "LAK", "LW", 2, "PP2", 0.20, 0.22, 0.14),
    ("Anze Kopitar",          "LAK", "C",  3, "PP2", 0.10, 0.26, 0.15),
    ("Drew Doughty",          "LAK", "D",  0, "PP1", 0.06, 0.26, 0.18),
    ("Sean Durzi",            "LAK", "D",  0, "PP2", 0.08, 0.22, 0.14),
    ("Phillip Danault",       "LAK", "C",  3, "-",   0.10, 0.18, 0.06),
    ("Mikey Anderson",        "LAK", "D",  0, "-",   0.03, 0.10, 0.03),
    ("Alex Laferriere",       "LAK", "LW", 3, "-",   0.10, 0.12, 0.05),
    ("Carl Grundstrom",       "LAK", "RW", 3, "-",   0.08, 0.10, 0.04),
    ("Tobias Bjornfot",       "LAK", "D",  0, "-",   0.04, 0.12, 0.04),

    # ── WASHINGTON CAPITALS (WSH) — E#7/WC1 seed ─────────────────────────────
    # Ovechkin approaches or passes Gretzky record; Carlson anchors the PP.
    ("Alex Ovechkin",         "WSH", "LW", 1, "PP1", 0.30, 0.28, 0.28),
    ("Dylan Strome",          "WSH", "C",  1, "PP1", 0.18, 0.32, 0.22),
    ("Tom Wilson",            "WSH", "RW", 1, "PP2", 0.22, 0.25, 0.10),
    ("Connor McMichael",      "WSH", "C",  2, "PP2", 0.18, 0.24, 0.14),
    ("Aliaksei Protas",       "WSH", "C",  2, "-",   0.14, 0.20, 0.08),
    ("John Carlson",          "WSH", "D",  0, "PP1", 0.08, 0.28, 0.20),
    ("Martin Fehervary",      "WSH", "D",  0, "-",   0.04, 0.14, 0.05),
    ("Andrew Mangiapane",     "WSH", "LW", 2, "PP2", 0.14, 0.18, 0.10),
    ("Sonny Milano",          "WSH", "LW", 3, "-",   0.10, 0.14, 0.06),
    ("Nicolas Aube-Kubel",    "WSH", "RW", 3, "-",   0.08, 0.08, 0.03),
    ("Rasmus Sandin",         "WSH", "D",  0, "PP2", 0.06, 0.20, 0.12),

    # ── BUFFALO SABRES (BUF) — E#3 seed ──────────────────────────────────────
    # Tage Thompson anchors a young, offensively deep squad.
    # Rasmus Dahlin is one of the elite offensive defencemen in the league.
    ("Tage Thompson",         "BUF", "C",  1, "PP1", 0.36, 0.40, 0.34),
    ("JJ Peterka",            "BUF", "RW", 1, "PP1", 0.28, 0.32, 0.24),
    ("Dylan Cozens",          "BUF", "C",  2, "PP1", 0.22, 0.30, 0.20),
    ("Rasmus Dahlin",         "BUF", "D",  0, "PP1", 0.10, 0.44, 0.34),
    ("Alex Tuch",             "BUF", "LW", 1, "PP2", 0.20, 0.26, 0.14),
    ("Zach Benson",           "BUF", "LW", 2, "PP2", 0.18, 0.24, 0.14),
    ("Henri Jokiharju",       "BUF", "D",  0, "PP2", 0.06, 0.22, 0.12),
    ("Jason Zucker",          "BUF", "LW", 2, "-",   0.14, 0.18, 0.06),
    ("Peyton Krebs",          "BUF", "C",  3, "-",   0.12, 0.16, 0.05),
    ("Mattias Samuelsson",    "BUF", "D",  0, "-",   0.04, 0.14, 0.04),
    ("Victor Olofsson",       "BUF", "LW", 3, "-",   0.12, 0.14, 0.08),

    # ── MONTREAL CANADIENS (MTL) — E#4 seed ──────────────────────────────────
    # Caufield and Suzuki form one of the best young 1-2 combos in the East.
    # Slafkovsky emerged as a power-forward franchise piece.
    ("Cole Caufield",         "MTL", "RW", 1, "PP1", 0.34, 0.28, 0.26),
    ("Nick Suzuki",           "MTL", "C",  1, "PP1", 0.24, 0.40, 0.28),
    ("Juraj Slafkovsky",      "MTL", "LW", 1, "PP2", 0.22, 0.26, 0.14),
    ("Kirby Dach",            "MTL", "C",  2, "PP1", 0.16, 0.26, 0.16),
    ("Mike Matheson",         "MTL", "D",  0, "PP1", 0.10, 0.34, 0.24),
    ("Alex Newhook",          "MTL", "C",  2, "PP2", 0.16, 0.22, 0.12),
    ("Kaiden Guhle",          "MTL", "D",  0, "PP2", 0.08, 0.22, 0.12),
    ("Christian Dvorak",      "MTL", "C",  3, "-",   0.10, 0.14, 0.04),
    ("Joel Armia",            "MTL", "RW", 3, "-",   0.10, 0.10, 0.03),
    ("David Savard",          "MTL", "D",  0, "-",   0.02, 0.10, 0.03),
    ("Emil Heineman",         "MTL", "LW", 3, "-",   0.08, 0.10, 0.03),

    # ── NEW YORK ISLANDERS (NYI) — E#5 seed ──────────────────────────────────
    # Barzal and Horvat (8yr UFA 2023) lead the offence.
    # Noah Dobson is the engine of the power play.
    ("Mathew Barzal",         "NYI", "C",  1, "PP1", 0.22, 0.40, 0.28),
    ("Bo Horvat",             "NYI", "C",  2, "PP1", 0.24, 0.26, 0.18),
    ("Anders Lee",            "NYI", "LW", 1, "PP2", 0.22, 0.20, 0.14),
    ("Brock Nelson",          "NYI", "C",  1, "PP2", 0.22, 0.24, 0.16),
    ("Noah Dobson",           "NYI", "D",  0, "PP1", 0.10, 0.32, 0.22),
    ("Oliver Wahlstrom",      "NYI", "RW", 2, "PP2", 0.18, 0.20, 0.12),
    ("Ryan Pulock",           "NYI", "D",  0, "PP1", 0.08, 0.28, 0.18),
    ("Adam Pelech",           "NYI", "D",  0, "-",   0.03, 0.14, 0.04),
    ("Jean-Gabriel Pageau",   "NYI", "C",  3, "-",   0.14, 0.18, 0.07),
    ("Kyle Palmieri",         "NYI", "RW", 2, "PP1", 0.18, 0.16, 0.12),
    ("Cal Clutterbuck",       "NYI", "RW", 4, "-",   0.08, 0.08, 0.02),

    # ── PHILADELPHIA FLYERS (PHI) — E#6 seed ─────────────────────────────────
    # Konecny is the catalyst; Couturier provides two-way depth.
    # Cam York has grown into an elite offensive defenceman.
    ("Travis Konecny",        "PHI", "C",  1, "PP1", 0.28, 0.36, 0.26),
    ("Sean Couturier",        "PHI", "C",  1, "PP1", 0.18, 0.30, 0.20),
    ("Owen Tippett",          "PHI", "RW", 1, "PP1", 0.22, 0.24, 0.18),
    ("Joel Farabee",          "PHI", "LW", 2, "PP2", 0.18, 0.24, 0.15),
    ("Cam York",              "PHI", "D",  0, "PP1", 0.10, 0.32, 0.22),
    ("Travis Sanheim",        "PHI", "D",  0, "PP2", 0.08, 0.26, 0.16),
    ("Morgan Frost",          "PHI", "C",  2, "PP2", 0.16, 0.24, 0.14),
    ("Noah Cates",            "PHI", "C",  3, "-",   0.12, 0.16, 0.06),
    ("Scott Laughton",        "PHI", "C",  3, "-",   0.10, 0.14, 0.04),
    ("Nick Seeler",           "PHI", "D",  0, "-",   0.02, 0.08, 0.02),
    ("Garnet Hathaway",       "PHI", "RW", 4, "-",   0.06, 0.08, 0.02),

    # ── PITTSBURGH PENGUINS (PIT) — E#8/WC2 seed ─────────────────────────────
    # Crosby (38) and Malkin (39) still lead the Penguins on a deep run.
    # Letang anchors the power play from the blue line.
    ("Sidney Crosby",         "PIT", "C",  1, "PP1", 0.32, 0.48, 0.35),
    ("Evgeni Malkin",         "PIT", "C",  1, "PP1", 0.26, 0.38, 0.28),
    ("Rickard Rakell",        "PIT", "RW", 1, "PP1", 0.20, 0.26, 0.18),
    ("Kris Letang",           "PIT", "D",  0, "PP1", 0.08, 0.35, 0.24),
    ("Bryan Rust",            "PIT", "RW", 2, "PP2", 0.16, 0.18, 0.08),
    ("Reilly Smith",          "PIT", "RW", 2, "PP2", 0.16, 0.22, 0.13),
    ("Marcus Pettersson",     "PIT", "D",  0, "PP2", 0.06, 0.16, 0.08),
    ("Lars Eller",            "PIT", "C",  3, "-",   0.08, 0.12, 0.03),
    ("Noel Acciari",          "PIT", "C",  4, "-",   0.06, 0.06, 0.02),
    ("Matt Nieto",            "PIT", "LW", 3, "-",   0.10, 0.10, 0.03),
    ("Kevin Hayes",           "PIT", "C",  2, "-",   0.12, 0.18, 0.06),

    # ── MINNESOTA WILD (MIN) — W#3 seed ──────────────────────────────────────
    # Kaprizov is a top-5 NHL scorer; Boldy and Zuccarello provide elite depth.
    ("Kirill Kaprizov",       "MIN", "LW", 1, "PP1", 0.36, 0.42, 0.36),
    ("Mats Zuccarello",       "MIN", "RW", 1, "PP1", 0.18, 0.42, 0.28),
    ("Matt Boldy",            "MIN", "LW", 1, "PP1", 0.26, 0.30, 0.22),
    ("Joel Eriksson Ek",      "MIN", "C",  2, "PP2", 0.18, 0.22, 0.12),
    ("Calen Addison",         "MIN", "D",  0, "PP1", 0.10, 0.32, 0.22),
    ("Frederick Gaudreau",    "MIN", "C",  2, "-",   0.14, 0.20, 0.08),
    ("Jared Spurgeon",        "MIN", "D",  0, "PP2", 0.08, 0.26, 0.18),
    ("Jonas Brodin",          "MIN", "D",  0, "-",   0.04, 0.18, 0.06),
    ("Marcus Foligno",        "MIN", "LW", 3, "-",   0.10, 0.12, 0.03),
    ("Sam Steel",             "MIN", "C",  3, "-",   0.10, 0.14, 0.04),
    ("Ryan Reaves",           "MIN", "RW", 4, "-",   0.04, 0.04, 0.01),

    # ── ANAHEIM DUCKS (ANA) — W#8 seed ───────────────────────────────────────
    # Surprise playoff team; Zegras and Terry lead a young, skilled offence.
    # McTavish and Carlsson are the future franchise centres.
    ("Trevor Zegras",         "ANA", "C",  1, "PP1", 0.22, 0.34, 0.24),
    ("Troy Terry",            "ANA", "RW", 1, "PP1", 0.24, 0.28, 0.22),
    ("Leo Carlsson",          "ANA", "C",  2, "PP1", 0.20, 0.24, 0.16),
    ("Mason McTavish",        "ANA", "C",  1, "PP2", 0.22, 0.26, 0.18),
    ("Frank Vatrano",         "ANA", "LW", 2, "PP2", 0.18, 0.18, 0.10),
    ("Cam Fowler",            "ANA", "D",  0, "PP1", 0.08, 0.30, 0.20),
    ("Jamie Drysdale",        "ANA", "D",  0, "PP2", 0.08, 0.24, 0.16),
    ("Ryan Strome",           "ANA", "C",  2, "-",   0.14, 0.18, 0.06),
    ("Max Jones",             "ANA", "LW", 3, "-",   0.10, 0.10, 0.03),
    ("Radko Gudas",           "ANA", "D",  0, "-",   0.02, 0.08, 0.02),
    ("Isac Lundestrom",       "ANA", "C",  3, "-",   0.10, 0.12, 0.04),

    # ── UTAH MAMMOTH (UTA) — W#5 seed ────────────────────────────────────────
    # Keller is the franchise player; Sergachev (traded from TBL) runs the PP.
    # Cooley and Guenther are elite young contributors.
    ("Clayton Keller",        "UTA", "C",  1, "PP1", 0.28, 0.40, 0.30),
    ("Nick Schmaltz",         "UTA", "C",  2, "PP1", 0.22, 0.30, 0.20),
    ("Dylan Guenther",        "UTA", "RW", 1, "PP2", 0.24, 0.26, 0.18),
    ("Logan Cooley",          "UTA", "C",  2, "PP2", 0.18, 0.24, 0.14),
    ("Mikhail Sergachev",     "UTA", "D",  0, "PP1", 0.10, 0.34, 0.24),
    ("Lawson Crouse",         "UTA", "LW", 2, "-",   0.16, 0.16, 0.06),
    ("Josh Doan",             "UTA", "RW", 3, "-",   0.12, 0.14, 0.05),
    ("Alexander Kerfoot",     "UTA", "C",  3, "-",   0.10, 0.16, 0.06),
    ("Juuso Valimaki",        "UTA", "D",  0, "PP2", 0.06, 0.18, 0.10),
    ("Michael Kesselring",    "UTA", "D",  0, "-",   0.04, 0.14, 0.05),
    ("Nick Bjugstad",         "UTA", "C",  4, "-",   0.06, 0.06, 0.02),
]

# ═══════════════════════════════════════════════════════════════════════════════
# PPG MODEL
# ═══════════════════════════════════════════════════════════════════════════════

def ppg_model(gpg: float, apg: float, ppp_pg: float) -> float:
    """
    Expected fantasy points per game for a player.

    base      = gpg × GOAL_PTS  + apg × ASSIST_PTS
    pp_bonus  = ppp_pg × PPP_BONUS
    ot_bonus  = (gpg × OT_GOAL_BONUS + apg × OT_ASSIST_BONUS) × OT_RATE

    The OT_RATE factor captures that roughly 10 % of all playoff points
    carry an OT game-winner bonus.
    """
    base     = gpg * GOAL_PTS + apg * ASSIST_PTS
    pp_bonus = ppp_pg * PPP_BONUS
    ot_bonus = (gpg * OT_GOAL_BONUS + apg * OT_ASSIST_BONUS) * OT_RATE
    return base + pp_bonus + ot_bonus


# ═══════════════════════════════════════════════════════════════════════════════
# DRAFT SCORE
# ═══════════════════════════════════════════════════════════════════════════════

_AVG_CUP_PROB = 1.0 / len(TEAMS)   # 1/16 ≈ 0.0625


def team_strength_boost(team_code: str) -> float:
    """
    Multiplicative boost based on a team's implied Stanley Cup probability
    relative to the league average.  Clamped to ±TEAM_BOOST_CAP (±25 %).
    """
    cup_prob = TEAMS[team_code]["cup_prob"]
    raw      = (cup_prob - _AVG_CUP_PROB) / _AVG_CUP_PROB * TEAM_BOOST_FACTOR
    return max(-TEAM_BOOST_CAP, min(TEAM_BOOST_CAP, raw))


def draft_score(gpg: float, apg: float, ppp_pg: float, team_code: str) -> float:
    """Draft Score = PPG_model × EGP × (1 + team_strength_boost)."""
    ppg  = ppg_model(gpg, apg, ppp_pg)
    egp  = TEAM_EGP[team_code]["egp_total"]
    boost = team_strength_boost(team_code)
    return ppg * egp * (1.0 + boost)


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD MASTER DATAFRAME
# ═══════════════════════════════════════════════════════════════════════════════

def build_player_df() -> pd.DataFrame:
    rows = []
    for raw in _RAW_PLAYERS:
        name, team, pos, line, pp_unit, gpg, apg, ppp_pg = raw
        if pos == "G":           # goalies don't accumulate individual pts
            continue
        t    = TEAMS[team]
        egp  = TEAM_EGP[team]["egp_total"]
        ppg  = ppg_model(gpg, apg, ppp_pg)
        ds   = draft_score(gpg, apg, ppp_pg, team)
        boost = team_strength_boost(team)
        rows.append(
            dict(
                player    = name,
                team      = team,
                team_name = t["name"],
                conf      = t["conf"],
                pos       = pos,
                line      = line if pos != "D" else "D",
                pp_unit   = pp_unit,
                gpg       = round(gpg, 3),
                apg       = round(apg, 3),
                ppp_pg    = round(ppp_pg, 3),
                ppg_model = round(ppg, 3),
                egp       = round(egp, 2),
                cup_prob  = t["cup_prob"],
                ts_boost  = round(boost, 4),
                draft_score = round(ds, 2),
            )
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("draft_score", ascending=False).reset_index(drop=True)
    df.index += 1
    df.index.name = "rank"
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM STACKING ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def _approx_win_rate(team_code: str) -> float:
    """
    Approximate fraction of games played that the team wins, across all
    rounds (factoring in series they win vs. lose).
    """
    t = TEAMS[team_code]
    # Weighted average over 4 rounds of: p_win_series × high_win_rate + ...
    r_probs = [t["r1_win"], t["r2_win"], t["r3_win"], t["r4_win"]]
    p_reach = [1.0]
    for rw in r_probs[:-1]:
        p_reach.append(p_reach[-1] * rw)

    total_games  = 0.0
    total_wins   = 0.0
    for rnd, p_series in enumerate(r_probs):
        p_game   = _pgame_from_series_prob(p_series)
        eg       = _expected_series_games(p_game)
        games    = p_reach[rnd] * eg
        # When team wins series: 4 wins + loses eg_won-4 games
        # When team loses series: wins eg_lost-4 games + 4 losses
        # Approximate: wins_per_game ≈ p_game  (per-game win rate)
        wins     = games * p_game
        total_games += games
        total_wins  += wins

    return total_wins / total_games if total_games > 0 else 0.50


def team_stack_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every playoff team score the total fantasy value of:
      • top-3 skaters by draft score (stack picks)
      • expected team-Win + team-Shutout points across all rounds
    """
    records = []
    for code, t in TEAMS.items():
        team_df   = df[df["team"] == code]
        top3      = team_df.head(3)
        player_ds = top3["draft_score"].sum()

        egp      = TEAM_EGP[code]["egp_total"]
        win_rate = _approx_win_rate(code)
        team_win_pts = egp * win_rate * TEAM_WIN_PTS
        team_so_pts  = egp * t["so_rate"] * TEAM_SHUTOUT_PTS

        records.append(
            dict(
                team          = code,
                name          = t["name"],
                cup_prob      = t["cup_prob"],
                egp           = round(egp, 2),
                top3_players  = " / ".join(top3["player"].tolist()),
                top3_ds       = round(player_ds, 2),
                team_win_pts  = round(team_win_pts, 2),
                team_so_pts   = round(team_so_pts, 2),
                combined_score= round(player_ds + team_win_pts + team_so_pts, 2),
            )
        )

    return (
        pd.DataFrame(records)
        .sort_values("combined_score", ascending=False)
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DRAFT-WINDOW ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def draft_window_label(rank: int) -> str:
    if rank in EARLY_PICKS:
        return "early (R1-2)"
    if rank in MID_PICKS:
        return "mid   (R3-6)"
    return "late  (R7-11)"


def best_by_window(df: pd.DataFrame, top_n: int = 10) -> dict[str, pd.DataFrame]:
    """Return top-N players for each draft window."""
    cols = ["player", "team", "pos", "line", "pp_unit", "ppg_model", "egp", "draft_score"]
    return {
        "early": df[df.index.isin(EARLY_PICKS)].head(top_n)[cols],
        "mid":   df[df.index.isin(MID_PICKS)].head(top_n)[cols],
        "late":  df[df.index.isin(LATE_PICKS)].head(top_n)[cols],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SLEEPER IDENTIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

def find_sleepers(df: pd.DataFrame, rank_cutoff: int = 75, top_n: int = 15) -> pd.DataFrame:
    """
    Sleepers: ranked outside the top *rank_cutoff* but with
      • PP1 involvement, OR
      • team EGP ≥ 14.0 (deep-run team), OR
      • ppg_model ≥ 0.65 (above-average scorer on a deeper team)
    """
    late_df = df[df.index > rank_cutoff].copy()
    mask = (
        (late_df["pp_unit"] == "PP1")
        | (late_df["egp"] >= 14.0)
        | (late_df["ppg_model"] >= 0.65)
    )
    cols = ["player", "team", "pos", "line", "pp_unit", "ppg_model", "egp", "draft_score"]
    return late_df[mask][cols].head(top_n)


# ═══════════════════════════════════════════════════════════════════════════════
# TEAM EGP SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

def egp_summary() -> pd.DataFrame:
    rows = []
    for code, egp_data in TEAM_EGP.items():
        t = TEAMS[code]
        rows.append(
            dict(
                team     = code,
                name     = t["name"],
                seed     = t["seed"],
                cup_prob = t["cup_prob"],
                egp_r1   = round(egp_data["egp_r1"], 2),
                egp_r2   = round(egp_data["egp_r2"], 2),
                egp_r3   = round(egp_data["egp_r3"], 2),
                egp_r4   = round(egp_data["egp_r4"], 2),
                egp_total= round(egp_data["egp_total"], 2),
                p_r2     = round(egp_data["p_reach_r2"], 3),
                p_r3     = round(egp_data["p_reach_r3"], 3),
                p_r4     = round(egp_data["p_reach_r4"], 3),
            )
        )
    return (
        pd.DataFrame(rows)
        .sort_values("egp_total", ascending=False)
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════════

_SEP = "=" * 80


def _header(title: str) -> None:
    print(f"\n{_SEP}")
    print(f"  {title}")
    print(_SEP)


def print_report(df: pd.DataFrame, write_csv: bool = False) -> None:

    # ── 1. EGP table ───────────────────────────────────────────────────────────
    _header("TEAM EXPECTED GAMES PLAYED (EGP) — 2026 NHL PLAYOFFS")
    egp_df = egp_summary()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(egp_df.to_string(index=False))

    # ── 2. Top 200 rankings ────────────────────────────────────────────────────
    _header("TOP 200 PLAYERS BY DRAFT SCORE")
    top200_cols = [
        "player", "team", "pos", "line", "pp_unit",
        "ppg_model", "egp", "cup_prob", "draft_score",
    ]
    top200 = df.head(200)
    print(top200[top200_cols].to_string())

    # ── 3. Best players by draft window ────────────────────────────────────────
    windows = best_by_window(df, top_n=12)

    _header("BEST PLAYERS — EARLY ROUND PICKS (Draft Rounds 1-2, Picks 1-28)")
    print(windows["early"].to_string(index=False))

    _header("BEST PLAYERS — MID ROUND PICKS (Draft Rounds 3-6, Picks 29-84)")
    print(windows["mid"].to_string(index=False))

    _header("BEST PLAYERS — LATE ROUND PICKS (Draft Rounds 7-11, Picks 85-154)")
    print(windows["late"].to_string(index=False))

    # ── 4. Team stacking analysis ──────────────────────────────────────────────
    _header("BEST TEAMS TO STACK (top 3 players + team Win/Shutout value)")
    stack_df = team_stack_analysis(df)
    print(stack_df.to_string(index=False))

    print(f"\n{'─'*80}")
    print("  TOP 3 STACKING TARGETS")
    print(f"{'─'*80}")
    for _, row in stack_df.head(3).iterrows():
        print(
            f"  #{int(_)+1}  {row['name']:<28}  "
            f"EGP={row['egp']:.1f}  Cup%={row['cup_prob']*100:.0f}%  "
            f"Combined={row['combined_score']:.1f}\n"
            f"       Stack: {row['top3_players']}"
        )
        print()

    # ── 5. Sleepers ────────────────────────────────────────────────────────────
    _header("HIGH-VALUE SLEEPERS (ranked outside top 75, strong PP/EGP/scoring)")
    sleepers = find_sleepers(df)
    print(sleepers.to_string(index=False))

    # ── 6. Scoring model reminder ──────────────────────────────────────────────
    _header("SCORING SYSTEM & MODEL NOTES")
    print(
        f"  Goals         = {GOAL_PTS}  pts\n"
        f"  Assists       = {ASSIST_PTS}  pts\n"
        f"  PP point      = +{PPP_BONUS} bonus (stacks on G or A)\n"
        f"  OT goal       = +{OT_GOAL_BONUS}  bonus\n"
        f"  OT assist     = +{OT_ASSIST_BONUS} bonus\n"
        f"  Team Win      = {TEAM_WIN_PTS}  pts per game won\n"
        f"  Team Shutout  = {TEAM_SHUTOUT_PTS}  pts per shutout\n"
        f"\n"
        f"  PPG_model = gpg×{GOAL_PTS} + apg×{ASSIST_PTS} + ppp_pg×{PPP_BONUS} + (gpg×{OT_GOAL_BONUS}+apg×{OT_ASSIST_BONUS})×{OT_RATE}\n"
        f"  Draft Score   = PPG_model × EGP × (1 + team_strength_boost)\n"
        f"  EGP           = Σ_r P(reach round r) × E[games in best-of-7 series r]\n"
        f"  team_boost    = clamp((cup_prob − 1/16) / (1/16) × 0.15,  −0.15, +0.25)\n"
        f"\n"
        f"  Per-game rates are 60% regular-season + 40% career playoff (weighted).\n"
        f"  League: {NUM_FANTASY_TEAMS} teams · snake draft · {ROSTER_SIZE} roster spots per team\n"
    )

    # ── 7. CSV output ──────────────────────────────────────────────────────────
    if write_csv:
        out_path = "rankings.csv"
        df.to_csv(out_path)
        print(f"  Rankings written to {out_path}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="NHL Playoff Pool Draft Model — 2026"
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Write full rankings to rankings.csv in addition to stdout"
    )
    args = parser.parse_args()

    df = build_player_df()
    print_report(df, write_csv=args.csv)


if __name__ == "__main__":
    main()
