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
    # R1 matchups (confirmed): CAR(M1) vs OTT(WC2), TBL(A2) vs MTL(A3),
    #                          BUF(A1) vs BOS(WC1), PIT(M2) vs PHI(M3)
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
        r1_win=0.54, r2_win=0.50, r3_win=0.46, r4_win=0.42,
        gpg_for=3.20, so_rate=0.06,
    ),
    "MTL": dict(
        name="Montreal Canadiens",    conf="E", seed=4,
        cup_prob=0.04,
        r1_win=0.40, r2_win=0.46, r3_win=0.42, r4_win=0.38,
        gpg_for=2.95, so_rate=0.07,
    ),
    "OTT": dict(
        name="Ottawa Senators",       conf="E", seed=5,
        cup_prob=0.03,
        r1_win=0.33, r2_win=0.40, r3_win=0.37, r4_win=0.34,
        gpg_for=3.00, so_rate=0.07,
    ),
    "PHI": dict(
        name="Philadelphia Flyers",   conf="E", seed=6,
        cup_prob=0.05,
        r1_win=0.50, r2_win=0.47, r3_win=0.43, r4_win=0.39,
        gpg_for=2.95, so_rate=0.07,
    ),
    "BOS": dict(
        name="Boston Bruins",         conf="E", seed=7,
        cup_prob=0.07,
        r1_win=0.48, r2_win=0.50, r3_win=0.46, r4_win=0.42,
        gpg_for=3.10, so_rate=0.08,
    ),
    "PIT": dict(
        name="Pittsburgh Penguins",   conf="E", seed=8,
        cup_prob=0.02,
        r1_win=0.50, r2_win=0.44, r3_win=0.40, r4_win=0.37,
        gpg_for=2.90, so_rate=0.07,
    ),
    # ── WESTERN CONFERENCE ──────────────────────────────────────────────────────
    # R1 matchups (confirmed): COL(C1) vs LAK(WC2), DAL(C2) vs MIN(C3),
    #                          VGK(P1) vs UTA(WC1), EDM(P2) vs ANA(P3)
    "COL": dict(
        name="Colorado Avalanche",    conf="W", seed=1,
        cup_prob=0.15,
        r1_win=0.68, r2_win=0.62, r3_win=0.56, r4_win=0.52,
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
        r1_win=0.68, r2_win=0.58, r3_win=0.53, r4_win=0.49,
        gpg_for=3.45, so_rate=0.06,
    ),
    "LAK": dict(
        name="Los Angeles Kings",     conf="W", seed=7,
        cup_prob=0.02,
        r1_win=0.32, r2_win=0.42, r3_win=0.38, r4_win=0.35,
        gpg_for=2.95, so_rate=0.10,
    ),
    "ANA": dict(
        name="Anaheim Ducks",         conf="W", seed=8,
        cup_prob=0.01,
        r1_win=0.33, r2_win=0.39, r3_win=0.36, r4_win=0.33,
        gpg_for=2.95, so_rate=0.07,
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

    # ── EDMONTON OILERS (EDM) — W/P2, R1 vs ANA ──────────────────────────────
    # Stats × 0.70 adjustment (60 % RS + 40 % career playoff blend).
    # McDavid/Draisaitl unchanged. McLeod traded to BUF; Savoie/Roslovic step up.
    ("Connor McDavid",        "EDM", "C",  1, "PP1", 0.415, 0.744, 0.449),
    ("Leon Draisaitl",        "EDM", "C",  1, "PP1", 0.377, 0.669, 0.453),
    ("Evan Bouchard",         "EDM", "D",  0, "PP1", 0.181, 0.614, 0.285),
    ("Ryan Nugent-Hopkins",   "EDM", "C",  2, "PP1", 0.187, 0.355, 0.276),
    ("Zach Hyman",            "EDM", "LW", 1, "PP2", 0.381, 0.246, 0.172),
    ("Mattias Ekholm",        "EDM", "D",  0, "-",   0.052, 0.302, 0.026),
    ("Vasily Podkolzin",      "EDM", "RW", 3, "PP2", 0.164, 0.156, 0.026),
    ("Jack Roslovic",         "EDM", "C",  3, "-",   0.216, 0.154, 0.062),
    ("Matt Savoie",           "EDM", "C",  3, "-",   0.130, 0.165, 0.052),

    # ── COLORADO AVALANCHE (COL) — W/C1, R1 vs LAK ───────────────────────────
    # Rantanen traded COL→CAR→DAL; Necas arrived from CAR.
    # Brock Nelson (from NYI) & Brent Burns (from CAR) bolster the roster.
    ("Nathan MacKinnon",      "COL", "C",  1, "PP1", 0.464, 0.648, 0.263),
    ("Martin Necas",          "COL", "RW", 1, "PP1", 0.341, 0.557, 0.215),
    ("Cale Makar",            "COL", "D",  0, "PP1", 0.189, 0.549, 0.274),
    ("Brock Nelson",          "COL", "C",  2, "PP1", 0.289, 0.280, 0.158),
    ("Gabriel Landeskog",     "COL", "LW", 2, "PP2", 0.163, 0.245, 0.058),
    ("Valeri Nichushkin",     "COL", "RW", 2, "PP2", 0.167, 0.296, 0.039),
    ("Artturi Lehkonen",      "COL", "LW", 2, "-",   0.213, 0.274, 0.020),
    ("Nazem Kadri",           "COL", "C",  3, "PP2", 0.145, 0.309, 0.173),
    ("Brent Burns",           "COL", "D",  0, "PP2", 0.104, 0.190, 0.009),
    ("Sam Malinski",          "COL", "D",  0, "PP2", 0.069, 0.277, 0.009),
    ("Parker Kelly",          "COL", "LW", 4, "-",   0.173, 0.121, 0.000),
    ("Josh Manson",           "COL", "D",  0, "-",   0.044, 0.230, 0.000),

    # ── TAMPA BAY LIGHTNING (TBL) — E/A2, R1 vs MTL ──────────────────────────
    # Stamkos (UFA, NSH); Sergachev traded to UTA.
    # Guentzel acquired at 2024 TDL; Raddysh now anchors the PP blue line.
    ("Nikita Kucherov",       "TBL", "RW", 1, "PP1", 0.405, 0.792, 0.341),
    ("Jake Guentzel",         "TBL", "LW", 1, "PP1", 0.328, 0.432, 0.259),
    ("Brandon Hagel",         "TBL", "LW", 2, "PP1", 0.355, 0.375, 0.118),
    ("Brayden Point",         "TBL", "C",  1, "PP1", 0.200, 0.355, 0.123),
    ("Darren Raddysh",        "TBL", "D",  0, "PP1", 0.211, 0.460, 0.249),
    ("Anthony Cirelli",       "TBL", "C",  2, "-",   0.227, 0.286, 0.039),
    ("Corey Perry",           "TBL", "RW", 3, "PP2", 0.165, 0.194, 0.127),
    ("Gage Goncalves",        "TBL", "C",  3, "-",   0.104, 0.208, 0.019),
    ("Oliver Bjorkstrand",    "TBL", "RW", 3, "PP2", 0.105, 0.175, 0.123),

    # ── VEGAS GOLDEN KNIGHTS (VGK) — W/P1, R1 vs UTA ────────────────────────
    # Marner acquired (TOR trade); Dorofeyev emerged as a 30-goal scorer.
    ("Jack Eichel",           "VGK", "C",  1, "PP1", 0.255, 0.596, 0.265),
    ("Mitch Marner",          "VGK", "RW", 1, "PP1", 0.207, 0.484, 0.207),
    ("Mark Stone",            "VGK", "RW", 1, "PP1", 0.327, 0.525, 0.315),
    ("Pavel Dorofeyev",       "VGK", "LW", 2, "PP1", 0.316, 0.230, 0.256),
    ("Ivan Barbashev",        "VGK", "LW", 2, "PP2", 0.196, 0.324, 0.051),
    ("Tomas Hertl",           "VGK", "C",  2, "PP2", 0.205, 0.290, 0.214),
    ("Rasmus Andersson",      "VGK", "D",  0, "PP2", 0.147, 0.259, 0.112),
    ("Shea Theodore",         "VGK", "D",  0, "PP1", 0.100, 0.290, 0.050),

    # ── CAROLINA HURRICANES (CAR) — E/M1, R1 vs OTT ──────────────────────────
    # Rantanen traded to DAL; Stankoven (DAL) + Ehlers (WPG) arrive in return.
    # Gostisbehere runs the PP blue line; Nikishin/Miller add defensive depth.
    ("Sebastian Aho",         "CAR", "C",  1, "PP1", 0.239, 0.469, 0.239),
    ("Nikolaj Ehlers",        "CAR", "LW", 1, "PP1", 0.222, 0.384, 0.248),
    ("Andrei Svechnikov",     "CAR", "RW", 2, "PP1", 0.275, 0.346, 0.257),
    ("Seth Jarvis",           "CAR", "C",  2, "PP2", 0.315, 0.335, 0.207),
    ("Jackson Blake",         "CAR", "RW", 3, "PP2", 0.190, 0.268, 0.104),
    ("Logan Stankoven",       "CAR", "C",  3, "-",   0.181, 0.199, 0.078),
    ("Shayne Gostisbehere",   "CAR", "D",  0, "PP1", 0.165, 0.471, 0.229),
    ("Taylor Hall",           "CAR", "LW", 3, "-",   0.158, 0.263, 0.061),
    ("Jordan Staal",          "CAR", "C",  4, "-",   0.187, 0.149, 0.037),
    ("K'Andre Miller",        "CAR", "D",  0, "PP2", 0.078, 0.282, 0.029),
    ("Alexander Nikishin",    "CAR", "D",  0, "PP2", 0.095, 0.190, 0.086),
    ("Sean Walker",           "CAR", "D",  0, "-",   0.078, 0.190, 0.009),

    # ── DALLAS STARS (DAL) — W/C2, R1 vs MIN ─────────────────────────────────
    # Rantanen acquired from CAR; Johnston & Robertson form a lethal 1-2 punch.
    ("Jason Robertson",       "DAL", "LW", 1, "PP1", 0.384, 0.435, 0.350),
    ("Wyatt Johnston",        "DAL", "C",  1, "PP1", 0.384, 0.350, 0.358),
    ("Mikko Rantanen",        "DAL", "RW", 1, "PP1", 0.241, 0.601, 0.372),
    ("Miro Heiskanen",        "DAL", "D",  0, "PP1", 0.082, 0.490, 0.255),
    ("Matt Duchene",          "DAL", "C",  2, "PP1", 0.196, 0.356, 0.172),
    ("Roope Hintz",           "DAL", "C",  2, "PP2", 0.198, 0.383, 0.251),
    ("Mavrik Bourque",        "DAL", "C",  3, "PP2", 0.171, 0.179, 0.043),
    ("Jamie Benn",            "DAL", "LW", 3, "-",   0.175, 0.245, 0.070),
    ("Michael Bunting",       "DAL", "LW", 3, "-",   0.132, 0.180, 0.095),
    ("Sam Steel",             "DAL", "C",  4, "-",   0.115, 0.201, 0.029),
    ("Thomas Harley",         "DAL", "D",  0, "PP2", 0.060, 0.300, 0.070),
    ("Esa Lindell",           "DAL", "D",  0, "-",   0.051, 0.222, 0.017),
    ("Justin Hryckowian",     "DAL", "C",  4, "-",   0.121, 0.139, 0.043),

    # ── LOS ANGELES KINGS (LAK) — W/WC2, R1 vs COL ───────────────────────────
    # Panarin acquired from NYR; Clarke emerges as offensive D anchor.
    ("Artemi Panarin",        "LAK", "LW", 1, "PP1", 0.255, 0.509, 0.209),
    ("Adrian Kempe",          "LAK", "RW", 1, "PP1", 0.315, 0.324, 0.105),
    ("Quinton Byfield",       "LAK", "C",  2, "PP1", 0.206, 0.224, 0.099),
    ("Kevin Fiala",           "LAK", "LW", 2, "PP2", 0.225, 0.275, 0.213),
    ("Alex Laferriere",       "LAK", "RW", 3, "-",   0.181, 0.190, 0.043),
    ("Brandt Clarke",         "LAK", "D",  0, "PP1", 0.069, 0.276, 0.112),
    ("Anze Kopitar",          "LAK", "C",  3, "-",   0.127, 0.276, 0.127),
    ("Trevor Moore",          "LAK", "LW", 3, "-",   0.134, 0.185, 0.010),

    # ── BOSTON BRUINS (BOS) — E/WC1, R1 vs BUF ───────────────────────────────
    # Pastrnak leads; McAvoy anchors the blue line.
    # Geekie had a breakout 39-goal season; Zacha is a reliable second-line C.
    ("David Pastrnak",        "BOS", "RW", 1, "PP1", 0.264, 0.646, 0.300),
    ("Charlie McAvoy",        "BOS", "D",  0, "PP1", 0.112, 0.508, 0.233),
    ("Pavel Zacha",           "BOS", "C",  1, "PP1", 0.269, 0.314, 0.197),
    ("Morgan Geekie",         "BOS", "C",  2, "PP2", 0.337, 0.251, 0.207),
    ("Viktor Arvidsson",      "BOS", "LW", 2, "PP1", 0.254, 0.295, 0.091),
    ("Elias Lindholm",        "BOS", "C",  2, "PP2", 0.172, 0.315, 0.203),
    ("Casey Mittelstadt",     "BOS", "C",  3, "PP2", 0.148, 0.266, 0.069),
    ("Fraser Minten",         "BOS", "C",  3, "-",   0.145, 0.154, 0.026),
    ("Marat Khusnutdinov",    "BOS", "C",  4, "-",   0.136, 0.164, 0.009),

    # ── BUFFALO SABRES (BUF) — E/A1, R1 vs BOS ───────────────────────────────
    # Peterka traded to UTA; Cozens traded to OTT; McLeod (EDM) & Doan (UTA) in.
    # Bowen Byram (from COL) bolsters the blue line alongside Dahlin.
    ("Tage Thompson",         "BUF", "C",  1, "PP1", 0.346, 0.354, 0.207),
    ("Rasmus Dahlin",         "BUF", "D",  0, "PP1", 0.173, 0.500, 0.200),
    ("Alex Tuch",             "BUF", "RW", 1, "PP2", 0.293, 0.293, 0.080),
    ("Jack Quinn",            "BUF", "RW", 2, "PP1", 0.171, 0.264, 0.094),
    ("Ryan McLeod",           "BUF", "C",  2, "PP2", 0.121, 0.346, 0.060),
    ("Zach Benson",           "BUF", "LW", 2, "PP2", 0.140, 0.323, 0.054),
    ("Jason Zucker",          "BUF", "LW", 3, "-",   0.271, 0.237, 0.181),
    ("Josh Doan",             "BUF", "RW", 3, "PP2", 0.213, 0.230, 0.145),
    ("Bowen Byram",           "BUF", "D",  0, "PP2", 0.094, 0.264, 0.060),
    ("Mattias Samuelsson",    "BUF", "D",  0, "-",   0.117, 0.251, 0.000),
    ("Peyton Krebs",          "BUF", "C",  3, "-",   0.102, 0.230, 0.009),
    ("Josh Norris",           "BUF", "C",  2, "PP2", 0.207, 0.334, 0.143),

    # ── MONTREAL CANADIENS (MTL) — E/A3, R1 vs TBL ───────────────────────────
    # Caufield/Suzuki/Hutson now joined by Dobson (from NYI) and Demidov.
    # Slafkovsky is a physical franchise presence; Kapanen provides depth.
    ("Cole Caufield",         "MTL", "RW", 1, "PP1", 0.441, 0.320, 0.251),
    ("Nick Suzuki",           "MTL", "C",  1, "PP1", 0.248, 0.615, 0.366),
    ("Lane Hutson",           "MTL", "D",  0, "PP1", 0.102, 0.564, 0.171),
    ("Juraj Slafkovsky",      "MTL", "LW", 1, "PP2", 0.256, 0.366, 0.239),
    ("Ivan Demidov",          "MTL", "RW", 2, "PP1", 0.162, 0.366, 0.171),
    ("Noah Dobson",           "MTL", "D",  0, "PP2", 0.105, 0.306, 0.061),
    ("Zachary Bolduc",        "MTL", "LW", 3, "-",   0.108, 0.162, 0.054),
    ("Mike Matheson",         "MTL", "D",  0, "-",   0.063, 0.269, 0.009),
    ("Oliver Kapanen",        "MTL", "C",  3, "-",   0.188, 0.128, 0.009),

    # ── OTTAWA SENATORS (OTT) — E/WC2, R1 vs CAR ────────────────────────────
    # Cozens (from BUF) joins Stutzle & Tkachuk as a dangerous 1-2-3 punch.
    # Sanderson leads the PP blue line; Batherson is a consistent 30-goal scorer.
    ("Tim Stutzle",           "OTT", "C",  1, "PP1", 0.298, 0.429, 0.254),
    ("Brady Tkachuk",         "OTT", "LW", 1, "PP1", 0.257, 0.432, 0.233),
    ("Drake Batherson",       "OTT", "RW", 1, "PP1", 0.293, 0.337, 0.266),
    ("Dylan Cozens",          "OTT", "C",  2, "PP1", 0.239, 0.265, 0.248),
    ("Jake Sanderson",        "OTT", "D",  0, "PP1", 0.147, 0.418, 0.230),
    ("Claude Giroux",         "OTT", "RW", 2, "PP2", 0.119, 0.299, 0.111),
    ("Shane Pinto",           "OTT", "C",  2, "-",   0.223, 0.223, 0.049),
    ("Ridly Greig",           "OTT", "C",  3, "-",   0.118, 0.200, 0.027),
    ("Michael Amadio",        "OTT", "RW", 3, "-",   0.130, 0.173, 0.000),
    ("Fabian Zetterlund",     "OTT", "LW", 3, "PP2", 0.145, 0.136, 0.068),
    ("Thomas Chabot",         "OTT", "D",  0, "PP2", 0.086, 0.295, 0.086),
    ("Jordan Spence",         "OTT", "D",  0, "-",   0.067, 0.230, 0.029),
    ("Artem Zub",             "OTT", "D",  0, "-",   0.043, 0.216, 0.000),

    # ── PHILADELPHIA FLYERS (PHI) — E/M3, R1 vs PIT ──────────────────────────
    # Zegras acquired from ANA; Michkov is a dynamic young scorer.
    # Dvorak (from MTL) provides two-way depth down the middle.
    ("Travis Konecny",        "PHI", "RW", 1, "PP1", 0.245, 0.373, 0.127),
    ("Trevor Zegras",         "PHI", "C",  1, "PP1", 0.225, 0.354, 0.199),
    ("Owen Tippett",          "PHI", "RW", 2, "PP2", 0.242, 0.199, 0.060),
    ("Matvei Michkov",        "PHI", "RW", 2, "PP1", 0.173, 0.268, 0.104),
    ("Sean Couturier",        "PHI", "C",  2, "-",   0.108, 0.215, 0.009),
    ("Christian Dvorak",      "PHI", "C",  3, "-",   0.158, 0.289, 0.044),
    ("Noah Cates",            "PHI", "LW", 3, "-",   0.154, 0.248, 0.077),
    ("Travis Sanheim",        "PHI", "D",  0, "PP2", 0.095, 0.225, 0.026),
    ("Jamie Drysdale",        "PHI", "D",  0, "PP1", 0.072, 0.215, 0.081),

    # ── PITTSBURGH PENGUINS (PIT) — E/M2, R1 vs PHI ──────────────────────────
    # Crosby & Malkin still lead; Erik Karlsson (traded from SJS/OTT) anchors PP.
    # Mantha (38 goals from PIT) is a physical power-forward.
    ("Sidney Crosby",         "PIT", "C",  1, "PP1", 0.298, 0.463, 0.237),
    ("Evgeni Malkin",         "PIT", "C",  1, "PP1", 0.237, 0.525, 0.275),
    ("Erik Karlsson",         "PIT", "D",  0, "PP1", 0.140, 0.476, 0.243),
    ("Rickard Rakell",        "PIT", "RW", 1, "PP2", 0.280, 0.280, 0.187),
    ("Bryan Rust",            "PIT", "RW", 2, "PP2", 0.282, 0.350, 0.233),
    ("Anthony Mantha",        "PIT", "RW", 2, "PP1", 0.285, 0.268, 0.112),
    ("Kris Letang",           "PIT", "D",  0, "PP2", 0.028, 0.293, 0.095),
    ("Egor Chinakhov",        "PIT", "RW", 3, "PP2", 0.204, 0.204, 0.058),
    ("Thomas Novak",          "PIT", "C",  3, "-",   0.137, 0.222, 0.060),
    ("Ben Kindel",            "PIT", "C",  3, "-",   0.154, 0.164, 0.091),
    ("Justin Brazeau",        "PIT", "RW", 4, "-",   0.186, 0.186, 0.055),
    ("Connor Dewar",          "PIT", "C",  4, "-",   0.126, 0.143, 0.000),
    ("Ryan Shea",             "PIT", "D",  0, "PP2", 0.053, 0.254, 0.000),

    # ── MINNESOTA WILD (MIN) — W/C3, R1 vs DAL ───────────────────────────────
    # Quinn Hughes (acquired from VAN) is now the PP quarterback.
    # Kaprizov + Boldy form arguably the best left-side duo in the West.
    ("Kirill Kaprizov",       "MIN", "LW", 1, "PP1", 0.404, 0.394, 0.287),
    ("Matt Boldy",            "MIN", "LW", 1, "PP1", 0.387, 0.396, 0.276),
    ("Quinn Hughes",          "MIN", "D",  0, "PP1", 0.066, 0.653, 0.321),
    ("Mats Zuccarello",       "MIN", "RW", 1, "PP2", 0.178, 0.463, 0.249),
    ("Joel Eriksson Ek",      "MIN", "C",  2, "PP2", 0.190, 0.320, 0.160),
    ("Brock Faber",           "MIN", "D",  0, "PP2", 0.131, 0.315, 0.097),
    ("Marcus Johansson",      "MIN", "LW", 2, "-",   0.140, 0.317, 0.084),
    ("Vladimir Tarasenko",    "MIN", "RW", 3, "PP2", 0.215, 0.224, 0.121),
    ("Ryan Hartman",          "MIN", "RW", 3, "-",   0.212, 0.184, 0.064),
    ("Bobby Brink",           "MIN", "RW", 3, "-",   0.154, 0.154, 0.072),

    # ── ANAHEIM DUCKS (ANA) — W/P3, R1 vs EDM ────────────────────────────────
    # Kreider, Trouba, John Carlson acquired (all NYR/WSH trades).
    # Gauthier leads a surprising offence; Sennecke is a top prospect.
    ("Cutter Gauthier",       "ANA", "LW", 1, "PP1", 0.373, 0.261, 0.177),
    ("Leo Carlsson",          "ANA", "C",  1, "PP1", 0.294, 0.375, 0.172),
    ("Beckett Sennecke",      "ANA", "RW", 2, "PP1", 0.199, 0.320, 0.112),
    ("Troy Terry",            "ANA", "RW", 2, "PP2", 0.210, 0.443, 0.128),
    ("Mason McTavish",        "ANA", "C",  2, "PP2", 0.161, 0.218, 0.104),
    ("Chris Kreider",         "ANA", "LW", 2, "PP1", 0.208, 0.265, 0.161),
    ("John Carlson",          "ANA", "D",  0, "PP1", 0.140, 0.440, 0.140),
    ("Jackson LaCombe",       "ANA", "D",  0, "PP2", 0.078, 0.415, 0.147),
    ("Mikael Granlund",       "ANA", "C",  3, "-",   0.233, 0.233, 0.135),
    ("Ryan Poehling",         "ANA", "C",  3, "-",   0.104, 0.227, 0.009),
    ("Jacob Trouba",          "ANA", "D",  0, "-",   0.088, 0.210, 0.009),
    ("Alex Killorn",          "ANA", "LW", 3, "-",   0.121, 0.156, 0.043),

    # ── UTAH MAMMOTH (UTA) — W/WC1, R1 vs VGK ───────────────────────────────
    # Peterka acquired from BUF; Doan traded to BUF; Carcone fills depth.
    # Sergachev is the PP1 quarterback; Keller/Guenther are elite forwards.
    ("Clayton Keller",        "UTA", "C",  1, "PP1", 0.225, 0.519, 0.233),
    ("Dylan Guenther",        "UTA", "RW", 1, "PP1", 0.354, 0.293, 0.213),
    ("Nick Schmaltz",         "UTA", "C",  1, "PP2", 0.285, 0.354, 0.173),
    ("Logan Cooley",          "UTA", "C",  2, "PP2", 0.317, 0.251, 0.132),
    ("JJ Peterka",            "UTA", "RW", 2, "PP1", 0.216, 0.181, 0.043),
    ("Mikhail Sergachev",     "UTA", "D",  0, "PP1", 0.091, 0.445, 0.236),
    ("Lawson Crouse",         "UTA", "LW", 2, "-",   0.201, 0.175, 0.009),
    ("Michael Carcone",       "UTA", "LW", 3, "-",   0.135, 0.135, 0.036),
    ("John Marino",           "UTA", "D",  0, "-",   0.036, 0.275, 0.000),
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
