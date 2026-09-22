#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NRDCPI + GFN Integrated Model v23
==================================
Coupling + Hysteresis + Liquidity + Financial Network
+ Research pipeline (Sobol, calibration, topology, Monte Carlo, unit tests)
+ Visualization & export (Plotly, PNG/SVG/PDF, XLSX, HTML dashboard)

Запуск:
    # Основное приложение Streamlit
    streamlit run NRDCPI_GFN_v23.py

    # Исследовательский пайплайн (без Streamlit)
    python NRDCPI_GFN_v23.py --only unit_tests
    python NRDCPI_GFN_v23.py --only calibration
    python NRDCPI_GFN_v23.py --only topology
    python NRDCPI_GFN_v23.py --only sensitivity --n_samples 64
    python NRDCPI_GFN_v23.py --only monte_carlo --n_seeds 30
    python NRDCPI_GFN_v23.py --only dashboard
    python NRDCPI_GFN_v23.py --only export --preset double_column
    python NRDCPI_GFN_v23.py --only xlsx
    python NRDCPI_GFN_v23.py --fast --open      # полный пайплайн быстро
    python NRDCPI_GFN_v23.py                     # полный пайплайн
"""

import warnings
import json
import os
import sys
import time
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union, Literal
from dataclasses import dataclass, field, replace
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIG: GFN + HYSTERESIS + LIQUIDITY
# =============================================================================

@dataclass
class GFNConfig:
    initial_risk: float = 0.48
    crisis_threshold: float = 0.68
    base_drift_mean: float = 0.003
    base_drift_std: float = 0.002
    shock_peak: float = 0.035
    contagion_coeff: float = 0.008
    volatility_std: float = 0.008
    coupling_strength: float = 0.12
    polarization_boost: float = 0.25


@dataclass
class HysteresisConfig:
    enter_threshold: float = 0.68
    exit_threshold: float = 0.60
    crisis_kappa_multiplier: float = 2.5
    use_hysteresis: bool = True


@dataclass
class LiquidityConfig:
    base_liquidity: float = 0.72
    min_liquidity: float = 0.15
    max_liquidity: float = 0.95
    gfn_drain_rate: float = 0.55
    jump_prob: float = 0.04
    jump_size_mean: float = 0.12
    jump_size_std: float = 0.05
    recovery_rate: float = 0.03
    liquidity_to_escalation_weight: float = 0.25


class GFNRiskModel:
    def __init__(self, config: GFNConfig = None, seed: int = 42):
        self.config = config or GFNConfig()
        self.rng = np.random.default_rng(seed)
        self.current_risk = self.config.initial_risk
        self.history: List[float] = [self.current_risk]
        self.month = 0

    def step(self, total_months: int = 60, scenario_multiplier: float = 1.0) -> float:
        cfg = self.config
        base_drift = self.rng.normal(cfg.base_drift_mean, cfg.base_drift_std)
        shock = 0.0
        progress = self.month / max(total_months, 1)
        if progress > 0.15:
            shock_peak = cfg.shock_peak * scenario_multiplier
            shock = shock_peak * np.exp(-((progress - 0.50) ** 2) / (2 * 0.18 ** 2))
        contagion = cfg.contagion_coeff * max(0.0, self.current_risk - 0.48)
        volatility = self.rng.normal(0, cfg.volatility_std)
        delta = base_drift + shock + contagion + volatility
        self.current_risk = float(np.clip(self.current_risk + delta, 0.05, 0.98))
        self.history.append(self.current_risk)
        self.month += 1
        return self.current_risk

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_risk = self.config.initial_risk
        self.history = [self.current_risk]
        self.month = 0


class LiquidityShockModel:
    def __init__(self, config: LiquidityConfig = None, seed: int = 42):
        self.config = config or LiquidityConfig()
        self.rng = np.random.default_rng(seed)
        self.liquidity = self.config.base_liquidity
        self.history: List[float] = [self.liquidity]
        self.shock_flags: List[int] = [0]

    def step(self, gfn_risk: float) -> Dict[str, float]:
        cfg = self.config
        drain = cfg.gfn_drain_rate * max(0.0, gfn_risk - 0.45) * 0.08
        jump, shock_flag = 0.0, 0
        if self.rng.random() < cfg.jump_prob:
            jump = abs(self.rng.normal(cfg.jump_size_mean, cfg.jump_size_std))
            shock_flag = 1
        recovery = 0.0
        if gfn_risk < 0.55 and self.liquidity < cfg.base_liquidity:
            recovery = cfg.recovery_rate * (cfg.base_liquidity - self.liquidity)
        self.liquidity = float(np.clip(
            self.liquidity - drain - jump + recovery,
            cfg.min_liquidity, cfg.max_liquidity
        ))
        self.history.append(self.liquidity)
        self.shock_flags.append(shock_flag)
        credit_factor = 0.3 + 0.7 * self.liquidity
        liq_escalation_push = (1.0 - self.liquidity) * cfg.liquidity_to_escalation_weight
        return {
            "liquidity": self.liquidity,
            "credit_factor": credit_factor,
            "liq_escalation_push": liq_escalation_push,
            "liquidity_shock": shock_flag,
        }

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.liquidity = self.config.base_liquidity
        self.history = [self.liquidity]
        self.shock_flags = [0]


# =============================================================================
# CONFIGS
# =============================================================================

@dataclass(frozen=True)
class AgentConfig:
    wealth_range: Tuple[float, float] = (12.0, 130.0)
    trust_range: Tuple[float, float] = (0.12, 0.88)
    risk_range: Tuple[float, float] = (0.08, 0.92)
    ideology_range: Tuple[float, float] = (-1.0, 1.0)
    innovation_range: Tuple[float, float] = (0.04, 0.96)
    adaptation_range: Tuple[float, float] = (0.04, 0.38)
    learning_range: Tuple[float, float] = (0.03, 0.28)
    confidence_range: Tuple[float, float] = (0.25, 0.65)
    type_distribution: Dict[str, float] = field(default_factory=lambda: {
        "household": 0.40, "firm": 0.22, "bank": 0.08,
        "government": 0.07, "foreign": 0.08, "innovation": 0.15
    })
    influence_rate_ideology: float = 0.40
    influence_rate_trust: float = 0.32
    risk_learning_rate: float = 0.08
    trust_learning_rate: float = 0.06
    payoff_memory: float = 0.70
    esc_perception_ideology_weight: float = 0.18
    esc_perception_local_weight: float = 0.10
    trust_macro_weight: float = 0.55
    trust_self_weight: float = 0.45
    payoff_threshold_high: float = 0.15
    payoff_threshold_low: float = -0.15


@dataclass(frozen=True)
class NetworkConfig:
    avg_degree: int = 8
    rewiring_probability: float = 0.12
    seed: int = 42


@dataclass(frozen=True)
class SDMConfig:
    territory_growth_rate: float = 0.42
    territory_investment_effect: float = 0.070
    territory_escalation_penalty: float = 0.078
    territory_ucdp_penalty: float = 0.016
    escalation_base_rate: float = 0.0130
    escalation_polarization_weight: float = 0.42
    escalation_tone_weight: float = 0.28
    escalation_sipri_weight: float = 0.12
    escalation_rand_weight: float = 0.12
    escalation_network_weight: float = 0.14
    escalation_trust_effect: float = 0.20
    escalation_innovation_effect: float = 0.085
    escalation_credit_effect: float = 0.045
    escalation_shock_effect: float = 0.45
    escalation_trust_recovery: float = 0.035
    escalation_trust_rate: float = 0.10
    nrdcpi_base: float = 43.5
    nrdcpi_territory_weight: float = 0.27
    nrdcpi_escalation_weight: float = 28.5
    nrdcpi_trust_weight: float = 9.5
    nrdcpi_polarization_weight: float = 14.5
    nrdcpi_tone_weight: float = 4.8
    nrdcpi_investment_weight: float = 4.7
    nrdcpi_tech_weight: float = 3.0
    nrdcpi_ucdp_weight: float = 6.5
    nrdcpi_sipri_weight: float = 3.5
    nrdcpi_vdem_weight: float = 2.8
    nrdcpi_shock_effect: float = 1.9
    nrdcpi_convergence_rate: float = 0.125
    trust_growth_rate: float = 0.017
    trust_escalation_penalty: float = 0.048
    trust_shock_effect: float = 0.28
    polarization_escalation_effect: float = 0.010
    polarization_tone_effect: float = 0.20
    polarization_innovation_effect: float = 0.0050
    polarization_ideology_effect: float = 0.30
    polarization_network_effect: float = 0.018
    pro_axis_peace_rate: float = 0.0040
    pro_axis_tech_effect: float = 0.09
    pro_axis_war_penalty: float = 0.010
    pro_axis_vdem_effect: float = 0.008
    territory_min: float = 12.0
    territory_max: float = 88.0
    escalation_min: float = 0.02
    escalation_max: float = 0.98
    nrdcpi_min: float = 14.0
    nrdcpi_max: float = 96.0
    trust_min: float = 0.08
    trust_max: float = 0.92
    polarization_min: float = 0.02
    polarization_max: float = 0.92
    pro_axis_min: float = 0.04
    pro_axis_max: float = 0.96
    dt: float = 0.025


@dataclass(frozen=True)
class KalmanConfig:
    initial_state: float = 47.0
    initial_covariance: float = 12.0
    process_noise: float = 2.2
    measurement_noise: float = 5.5


@dataclass(frozen=True)
class BayesianConfig:
    initial_alpha: float = 2.2
    initial_beta: float = 7.8
    escalation_threshold: float = 0.006


@dataclass(frozen=True)
class VSMConfig:
    escalation_weight: float = 0.8
    polarization_weight: float = 0.7
    tone_weight: float = 0.5
    pro_axis_scale: float = 2.0
    pro_axis_offset: float = 0.5
    nrdcpi_min_ref: float = 15.0
    nrdcpi_max_ref: float = 95.0


@dataclass(frozen=True)
class ExternalDataConfig:
    base_values: Dict[str, float] = field(default_factory=lambda: {
        "ucdp_intensity": 0.41, "sipri_ratio": 1.18,
        "vdem_libdem_avg": 0.39, "iiss_capability": 0.57,
        "rand_strategic": 0.49,
    })
    noise_std: float = 0.02
    use_noise: bool = True


@dataclass(frozen=True)
class ScenarioConfig:
    scenarios: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "base":       {"shock_mean": 0.0,    "shock_std": 0.0040, "trust_break": 1.0,  "gfn_mult": 1.0},
        "peace":      {"shock_mean": -0.014, "shock_std": 0.0030, "trust_break": 1.0,  "gfn_mult": 0.7},
        "war":        {"shock_mean": 0.020,  "shock_std": 0.0045, "trust_break": 1.0,  "gfn_mult": 1.3},
        "freeze":     {"shock_mean": 0.0025, "shock_std": 0.0020, "trust_break": 1.0,  "gfn_mult": 1.0},
        "nuclear":    {"shock_mean": 0.042,  "shock_std": 0.0075, "trust_break": 1.55, "gfn_mult": 1.8},
        "gfn_crisis": {"shock_mean": 0.015,  "shock_std": 0.0050, "trust_break": 1.2,  "gfn_mult": 2.2},
    })


@dataclass(frozen=True)
class ModelConfig:
    n_agents: int = 320
    random_seed: int = 42
    version: str = "23.0-GFN-Hysteresis-Liquidity-Research"
    use_network_abm: bool = True
    use_gfn_coupling: bool = True
    gfn: GFNConfig = field(default_factory=GFNConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    sdm: SDMConfig = field(default_factory=SDMConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)
    vsm: VSMConfig = field(default_factory=VSMConfig)
    external: ExternalDataConfig = field(default_factory=ExternalDataConfig)
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)


DEFAULT_CONFIG = ModelConfig()


# =============================================================================
# FILTERS / EXTERNAL / GDELT
# =============================================================================

class ExternalDataIntegrator:
    def __init__(self, config: ExternalDataConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self._base = config.base_values.copy()

    def get_all_external(self) -> Dict[str, float]:
        data = {}
        for k, v in self._base.items():
            if self.config.use_noise:
                data[k] = float(np.clip(v + self.rng.normal(0, self.config.noise_std), 0.01, 0.99))
            else:
                data[k] = float(v)
        return data


class GDELTDataLoader:
    def __init__(self):
        self.fallback = {
            "base": -0.30, "war": -0.45, "peace": 0.10,
            "freeze": -0.16, "nuclear": -0.52, "gfn_crisis": -0.40
        }

    def compute_visual_tone(self, scenario: str = "base") -> float:
        tone = self.fallback.get(scenario, -0.30)
        return float(np.clip(tone + np.random.normal(0, 0.03), -1.0, 1.0))


class KalmanFilter1D:
    def __init__(self, config: KalmanConfig):
        self.config = config
        self.x_est = config.initial_state
        self.P_est = config.initial_covariance

    def update(self, measurement: float) -> Dict[str, float]:
        x_pred = self.x_est
        P_pred = self.P_est + self.config.process_noise
        K = P_pred / (P_pred + self.config.measurement_noise)
        self.x_est = x_pred + K * (measurement - x_pred)
        self.P_est = (1 - K) * P_pred
        return {"estimate": self.x_est, "kalman_gain": K, "variance": self.P_est}


class BayesianBetaUpdater:
    def __init__(self, config: BayesianConfig):
        self.config = config
        self.alpha = config.initial_alpha
        self.beta_val = config.initial_beta

    def get_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta_val)

    def update_from_escalation_change(self, delta_esc: float):
        th = self.config.escalation_threshold
        if delta_esc > th:
            self.alpha += 1
        elif delta_esc < -th:
            self.beta_val += 1


class VSM:
    def __init__(self, config: VSMConfig):
        self.config = config

    def diagnose(self, state: Dict[str, float]) -> Dict[str, Any]:
        cfg = self.config
        s1 = float(np.clip(1 - state.get("escalation", 0.5) * cfg.escalation_weight, 0, 1))
        s2 = float(np.clip(1 - state.get("polarization", 0.3) * cfg.polarization_weight, 0, 1))
        s3 = float(np.clip(1 - abs(state.get("visual_tone", 0)) * cfg.tone_weight, 0, 1))
        pro = state.get("pro_axis_ratio", 0.5)
        s4 = float(np.clip(pro * cfg.pro_axis_scale - cfg.pro_axis_offset, 0, 1))
        nrd = state.get("nrdcpi", 50)
        s5 = float(np.clip(1 - (nrd - cfg.nrdcpi_min_ref) / (cfg.nrdcpi_max_ref - cfg.nrdcpi_min_ref) * 0.5, 0, 1))
        scores = {1: s1, 2: s2, 3: s3, 4: s4, 5: s5}
        return {"scores": scores, "viability": float(np.mean(list(scores.values())))}


# =============================================================================
# ABM
# =============================================================================

class Agent:
    def __init__(self, agent_type: str, agent_id: int, config: AgentConfig, rng: np.random.Generator):
        self.type = agent_type
        self.id = agent_id
        self._config = config
        self._rng = rng
        self.wealth = rng.uniform(*config.wealth_range)
        self.trust = rng.uniform(*config.trust_range)
        self.risk = rng.uniform(*config.risk_range)
        self.ideology = rng.uniform(*config.ideology_range)
        self.innovation = rng.uniform(*config.innovation_range)
        self.adaptation_rate = rng.uniform(*config.adaptation_range)
        self.learning_rate = rng.uniform(*config.learning_range)
        self.bounded_confidence = rng.uniform(*config.confidence_range)
        self.recent_payoff = 0.0

    def sense_neighbours(self, neighbour_ids, agents):
        if not neighbour_ids:
            return {"avg_ideology": self.ideology, "avg_trust": self.trust, "n": 0}
        ides, trusts = [], []
        for nid in neighbour_ids:
            if 0 <= nid < len(agents):
                ides.append(agents[nid].ideology)
                trusts.append(agents[nid].trust)
        if not ides:
            return {"avg_ideology": self.ideology, "avg_trust": self.trust, "n": 0}
        return {"avg_ideology": float(np.mean(ides)), "avg_trust": float(np.mean(trusts)), "n": len(ides)}

    def social_influence(self, local):
        if local["n"] == 0:
            return
        d_i = local["avg_ideology"] - self.ideology
        if abs(d_i) < self.bounded_confidence:
            self.ideology = float(np.clip(
                self.ideology + self.adaptation_rate * self._config.influence_rate_ideology * d_i, -1, 1))
        d_t = local["avg_trust"] - self.trust
        if abs(d_t) < self.bounded_confidence * 1.2:
            self.trust = float(np.clip(
                self.trust + self.adaptation_rate * self._config.influence_rate_trust * d_t, 0.05, 0.95))

    def individual_learning(self, macro_esc, macro_trust):
        payoff = -0.6 * macro_esc + 0.4 * macro_trust + self._rng.normal(0, 0.05)
        self.recent_payoff = (self._config.payoff_memory * self.recent_payoff +
                              (1 - self._config.payoff_memory) * payoff)
        if self.recent_payoff < self._config.payoff_threshold_low:
            self.risk = float(np.clip(self.risk - self.learning_rate * self._config.risk_learning_rate, 0.05, 0.95))
            self.trust = float(np.clip(self.trust - self.learning_rate * self._config.trust_learning_rate, 0.05, 0.95))
        elif self.recent_payoff > self._config.payoff_threshold_high:
            self.risk = float(np.clip(self.risk + self.learning_rate * 0.05, 0.05, 0.95))
            self.trust = float(np.clip(self.trust + self.learning_rate * 0.04, 0.05, 0.95))

    def decide(self, context, local):
        esc = context.get("escalation", 0.4)
        terr = context.get("territory", 50.0)
        trust_m = context.get("trust", 0.5)
        tech = context.get("tech_access", 0.4)
        external = context.get("external", {})
        local_ideo = local.get("avg_ideology", self.ideology)
        perc_esc = esc * (1 + self._config.esc_perception_ideology_weight * abs(self.ideology) +
                          self._config.esc_perception_local_weight * abs(local_ideo))
        perc_trust = self._config.trust_macro_weight * trust_m + self._config.trust_self_weight * self.trust

        if self.type == "household":
            save = np.clip(0.17 + 0.18 * perc_esc - 0.06 * (terr / 100) + 0.05 * self.risk, 0.05, 0.65)
            cons = self.wealth * (1 - save)
            return {
                "consumption": cons, "aggregate_demand": cons * 0.25,
                "trust_change": -perc_esc * 0.030 * (1 - self.adaptation_rate),
                "ideology_shift": 0.010 * np.sign(self.ideology) * perc_esc
            }
        if self.type == "firm":
            inv = np.clip(0.11 * perc_trust * (1 - perc_esc * 0.34) *
                          (0.75 + 0.45 * self.innovation), 0.025, 0.45)
            return {"investment": inv, "innovation_spending": self.innovation * 0.100 * (1 - perc_esc * 0.15)}
        if self.type == "bank":
            return {"credit_availability": max(0.20, 1 - perc_esc * 0.32 - 0.11 * (1 - perc_trust))}
        if self.type == "government":
            sipri = external.get("sipri_ratio", 1.15)
            return {"military_spending_multiplier": 1 + perc_esc * 0.28 * (1 + 0.10 * sipri)}
        if self.type == "foreign":
            return {
                "investment_inflow": 0.030 * perc_trust * (1 - perc_esc * 0.36),
                "sanctions_pressure": perc_esc * 0.16
            }
        return {"tech_transfer": tech * 0.045 * (1 + 0.22 * local.get("n", 0) / 10.0)}


class AgentPopulation:
    def __init__(self, config: ModelConfig, rng=None):
        self.config = config
        self.rng = rng or np.random.default_rng(config.random_seed)
        self.agents = []
        self.network = nx.Graph()
        self.aggregated = {}
        self.network_stats = {}
        self._init_pop()
        if config.use_network_abm:
            self._build_net()

    def _init_pop(self):
        dist = self.config.agent.type_distribution
        n = self.config.n_agents
        for t, p in dist.items():
            for _ in range(int(n * p)):
                self.agents.append(Agent(t, len(self.agents), self.config.agent, self.rng))
        while len(self.agents) < n:
            self.agents.append(Agent("household", len(self.agents), self.config.agent, self.rng))

    def _build_net(self):
        n = len(self.agents)
        nc = self.config.network
        self.network = nx.watts_strogatz_graph(n, nc.avg_degree, nc.rewiring_probability, seed=nc.seed)

    def step(self, context):
        if self.config.use_network_abm and self.network.number_of_nodes() > 0:
            for i, a in enumerate(self.agents):
                local = a.sense_neighbours(list(self.network.neighbors(i)), self.agents)
                a.social_influence(local)
                a.individual_learning(context.get("escalation", 0.4), context.get("trust", 0.5))
        results = defaultdict(float)
        for i, a in enumerate(self.agents):
            if self.config.use_network_abm and self.network.number_of_nodes() > 0:
                local = a.sense_neighbours(list(self.network.neighbors(i)), self.agents)
            else:
                local = {"avg_ideology": a.ideology, "avg_trust": a.trust, "n": 0}
            for k, v in a.decide(context, local).items():
                results[k] += v
        n = len(self.agents)
        for k in results:
            if k != "military_spending_multiplier":
                results[k] /= n
        self.aggregated = dict(results)
        self._update_stats()
        return self.aggregated

    def _update_stats(self):
        ides = [a.ideology for a in self.agents]
        self.network_stats = {
            "avg_ideology": float(np.mean(ides)),
            "polarization_proxy": float(np.mean(np.abs(ides))),
            "network_density": float(nx.density(self.network)) if self.network.number_of_nodes() else 0.0,
        }

    def get_feedback_factors(self):
        d = self.aggregated
        return {
            "investment_factor": 1 + (d.get("investment", 0.11) - 0.11) * 0.60,
            "innovation_factor": 1 + d.get("innovation_spending", 0.05) * 2.0,
            "trust_factor": 1 + d.get("trust_change", 0) * 0.70,
            "tech_factor": 1 + d.get("tech_transfer", 0.02) * 0.80,
            "credit_availability": d.get("credit_availability", 0.7),
            "network_polarization": self.network_stats.get("polarization_proxy", 0.4),
            "ideology_pressure": d.get("ideology_shift", 0) * 1.5,
            "military_factor": d.get("military_spending_multiplier", 1.0),
        }

    def get_network_for_viz(self):
        return self.network, [a.ideology for a in self.agents], [a.type for a in self.agents]


# =============================================================================
# SDM
# =============================================================================

def sdm_equations(state, t, params, config: SDMConfig):
    territory, escalation, nrdcpi_raw, trust, polarization, pro_axis = state
    abm = params.get("abm_factors", {})
    gdelt_tone = params.get("gdelt_tone", 0.0)
    external = params.get("external", {})
    external_shock = params.get("external_shock", 0.0)
    trust_break = params.get("trust_break_factor", 1.0)
    gfn_risk = params.get("gfn_risk", 0.48)
    gfn_coupling = params.get("gfn_coupling", 0.0)

    inv_f = abm.get("investment_factor", 1.0)
    innov_f = abm.get("innovation_factor", 1.0)
    trust_f = abm.get("trust_factor", 1.0)
    tech_f = abm.get("tech_factor", 1.0)
    credit = abm.get("credit_availability", 0.7)
    net_pol = abm.get("network_polarization", 0.4)

    ucdp = external.get("ucdp_intensity", 0.38)
    sipri_r = external.get("sipri_ratio", 1.15)
    vdem = external.get("vdem_libdem_avg", 0.39)
    rand_s = external.get("rand_strategic", 0.48)
    cfg = config

    gfn_esc_boost = gfn_coupling * max(0.0, gfn_risk - 0.45) * 1.8
    gfn_pol_boost = gfn_coupling * max(0.0, gfn_risk - 0.50) * 0.6

    d_terr = (cfg.territory_growth_rate * (1 - escalation) * (1 + inv_f * cfg.territory_investment_effect) -
              cfg.territory_escalation_penalty * escalation * (1 + external_shock) -
              cfg.territory_ucdp_penalty * ucdp)

    d_esc = (cfg.escalation_base_rate *
             (1 + polarization * cfg.escalation_polarization_weight +
              max(0, -gdelt_tone) * cfg.escalation_tone_weight +
              (sipri_r - 1.0) * cfg.escalation_sipri_weight +
              rand_s * cfg.escalation_rand_weight +
              net_pol * cfg.escalation_network_weight +
              gfn_esc_boost) *
             (1 - trust * cfg.escalation_trust_effect -
              innov_f * cfg.escalation_innovation_effect -
              credit * cfg.escalation_credit_effect) +
             external_shock * cfg.escalation_shock_effect -
             cfg.escalation_trust_recovery * trust * cfg.escalation_trust_rate)

    base = (cfg.nrdcpi_base - territory * cfg.nrdcpi_territory_weight +
            escalation * cfg.nrdcpi_escalation_weight - trust * cfg.nrdcpi_trust_weight +
            polarization * cfg.nrdcpi_polarization_weight - gdelt_tone * cfg.nrdcpi_tone_weight +
            (1 - inv_f) * cfg.nrdcpi_investment_weight + (1 - tech_f) * cfg.nrdcpi_tech_weight +
            ucdp * cfg.nrdcpi_ucdp_weight + (sipri_r - 1) * cfg.nrdcpi_sipri_weight +
            (1 - vdem) * cfg.nrdcpi_vdem_weight)
    d_nrd = cfg.nrdcpi_convergence_rate * (base - nrdcpi_raw) + external_shock * cfg.nrdcpi_shock_effect

    d_trust = (cfg.trust_growth_rate * (1 - escalation) * (1 + trust_f * 0.10) * (0.65 + 0.35 * vdem) -
               cfg.trust_escalation_penalty * escalation *
               (1 + external_shock * cfg.trust_shock_effect) * trust_break)

    d_pol = (cfg.polarization_escalation_effect * escalation *
             (1 + abs(gdelt_tone) * cfg.polarization_tone_effect) -
             cfg.polarization_innovation_effect * (1 + innov_f * 0.10) +
             abm.get("ideology_pressure", 0) * cfg.polarization_ideology_effect +
             net_pol * cfg.polarization_network_effect +
             gfn_pol_boost)

    d_pro = (cfg.pro_axis_peace_rate * (1 - escalation) * (1 + tech_f * cfg.pro_axis_tech_effect) -
             cfg.pro_axis_war_penalty * escalation +
             cfg.pro_axis_vdem_effect * (1 - vdem))

    return np.array([d_terr, d_esc, d_nrd, d_trust, d_pol, d_pro])


def run_sdm_step(state, params, config: SDMConfig, months=1.0):
    dt = config.dt
    n_steps = int(months / dt)
    s = state.copy()
    for _ in range(n_steps):
        k1 = sdm_equations(s, 0, params, config)
        k2 = sdm_equations(s + 0.5 * dt * k1, 0.5 * dt, params, config)
        k3 = sdm_equations(s + 0.5 * dt * k2, 0.5 * dt, params, config)
        k4 = sdm_equations(s + dt * k3, dt, params, config)
        s = s + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        s[0] = np.clip(s[0], config.territory_min, config.territory_max)
        s[1] = np.clip(s[1], config.escalation_min, config.escalation_max)
        s[2] = np.clip(s[2], config.nrdcpi_min, config.nrdcpi_max)
        s[3] = np.clip(s[3], config.trust_min, config.trust_max)
        s[4] = np.clip(s[4], config.polarization_min, config.polarization_max)
        s[5] = np.clip(s[5], config.pro_axis_min, config.pro_axis_max)
    return s


# =============================================================================
# MAIN MODEL
# =============================================================================

class NRDCPI_GFN_v22:
    def __init__(self, config: ModelConfig = DEFAULT_CONFIG):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
        self.population = AgentPopulation(config, self.rng)
        self.kalman = KalmanFilter1D(config.kalman)
        self.bayesian = BayesianBetaUpdater(config.bayesian)
        self.vsm = VSM(config.vsm)
        self.gdelt = GDELTDataLoader()
        self.external = ExternalDataIntegrator(config.external, self.rng)
        self.gfn = GFNRiskModel(config.gfn, seed=config.random_seed)
        self.liquidity_model = LiquidityShockModel(LiquidityConfig(), seed=config.random_seed)
        self.hyst = HysteresisConfig()
        self.in_crisis_mode = False
        self._use_coupling = True
        self._coupling_strength = 0.12
        self.state = {
            "territory": 49.0, "escalation": 0.41, "nrdcpi": 46.5,
            "trust": 0.44, "polarization": 0.34, "pro_axis_ratio": 0.46,
            "visual_tone": -0.28, "nrdcpi_filtered": 46.5,
            "gfn_risk": config.gfn.initial_risk, "liquidity": 0.72
        }
        self.history: List[Dict] = []
        self.scenario = "base"

    def update_crisis_mode(self, gfn_risk: float) -> bool:
        if not self.hyst.use_hysteresis:
            self.in_crisis_mode = gfn_risk >= self.hyst.enter_threshold
            return self.in_crisis_mode
        if self.in_crisis_mode:
            if gfn_risk <= self.hyst.exit_threshold:
                self.in_crisis_mode = False
        else:
            if gfn_risk >= self.hyst.enter_threshold:
                self.in_crisis_mode = True
        return self.in_crisis_mode

    def step(self, scenario: str = "base", total_months: int = 36) -> Dict:
        self.scenario = scenario
        scen_p = self.config.scenario.scenarios.get(scenario, {})
        gfn_mult = scen_p.get("gfn_mult", 1.0)

        gfn_risk = self.gfn.step(total_months=total_months, scenario_multiplier=gfn_mult)
        self.state["gfn_risk"] = gfn_risk

        in_crisis = self.update_crisis_mode(gfn_risk)
        base_kappa = self._coupling_strength if self._use_coupling else 0.0
        eff_kappa = base_kappa * self.hyst.crisis_kappa_multiplier if in_crisis else base_kappa

        liq = self.liquidity_model.step(gfn_risk)
        self.state["liquidity"] = liq["liquidity"]

        last_ext = self.external.get_all_external()
        tone = self.gdelt.compute_visual_tone(scenario=scenario)
        self.state["visual_tone"] = tone

        shock_mean = scen_p.get("shock_mean", 0.0)
        shock_std = scen_p.get("shock_std", 0.004)
        trust_break = scen_p.get("trust_break", 1.0)
        external_shock = shock_mean + self.rng.normal(0, shock_std)

        gfn_extra_shock = eff_kappa * max(0.0, gfn_risk - 0.45) * 0.35
        external_shock += gfn_extra_shock + liq["liq_escalation_push"] * 0.4

        vsm_diag = self.vsm.diagnose({
            "escalation": self.state["escalation"],
            "polarization": self.state["polarization"],
            "pro_axis_ratio": self.state["pro_axis_ratio"],
            "visual_tone": tone,
            "nrdcpi": self.state["nrdcpi"]
        })

        abm_ctx = {
            "trust": self.state["trust"], "escalation": self.state["escalation"],
            "territory": self.state["territory"], "tech_access": 0.42, "external": last_ext
        }
        self.population.step(abm_ctx)
        abm_f = self.population.get_feedback_factors()
        abm_f["credit_availability"] = abm_f.get("credit_availability", 0.7) * liq["credit_factor"]

        sdm_params = {
            "external_shock": external_shock, "gdelt_tone": tone,
            "abm_factors": abm_f, "external": last_ext,
            "trust_break_factor": trust_break, "gfn_risk": gfn_risk,
            "gfn_coupling": eff_kappa,
        }
        old = np.array([
            self.state["territory"], self.state["escalation"], self.state["nrdcpi"],
            self.state["trust"], self.state["polarization"], self.state["pro_axis_ratio"]
        ])
        new = run_sdm_step(old, sdm_params, self.config.sdm)

        self.state["territory"] = float(new[0])
        self.state["escalation"] = float(new[1])
        self.state["nrdcpi"] = float(new[2])
        self.state["trust"] = float(new[3])
        self.state["polarization"] = float(new[4])
        self.state["pro_axis_ratio"] = float(new[5])

        kal = self.kalman.update(self.state["nrdcpi"])
        self.state["nrdcpi_filtered"] = kal["estimate"]
        self.bayesian.update_from_escalation_change(self.state["escalation"] - old[1])

        self.history.append({
            "month": len(self.history),
            "territory": round(self.state["territory"], 2),
            "escalation": round(self.state["escalation"], 4),
            "nrdcpi": round(self.state["nrdcpi"], 2),
            "nrdcpi_filtered": round(self.state["nrdcpi_filtered"], 2),
            "trust": round(self.state["trust"], 4),
            "polarization": round(self.state["polarization"], 4),
            "pro_axis_ratio": round(self.state["pro_axis_ratio"], 4),
            "visual_tone": round(tone, 4),
            "gfn_risk": round(gfn_risk, 4),
            "gfn_crisis": int(gfn_risk >= self.config.gfn.crisis_threshold),
            "in_crisis_mode": int(in_crisis),
            "eff_kappa": round(eff_kappa, 4),
            "liquidity": round(liq["liquidity"], 4),
            "credit_factor": round(liq["credit_factor"], 4),
            "liquidity_shock": liq["liquidity_shock"],
            "beta": round(self.bayesian.get_mean(), 4),
            "vsm_viability": round(vsm_diag["viability"], 3),
            "network_pol": round(self.population.network_stats.get("polarization_proxy", 0), 3),
            "external_shock": round(external_shock, 4),
            "gfn_extra_shock": round(gfn_extra_shock, 4),
        })
        return self.state

    def run_simulation(self, months: int = 36, scenario: str = "base",
                       use_coupling: bool = True, coupling_strength: float = 0.12,
                       use_hysteresis: bool = True,
                       crisis_multiplier: float = 2.5) -> pd.DataFrame:
        self.history = []
        self.state = {
            "territory": 49.0, "escalation": 0.41, "nrdcpi": 46.5,
            "trust": 0.44, "polarization": 0.34, "pro_axis_ratio": 0.46,
            "visual_tone": -0.28, "nrdcpi_filtered": 46.5,
            "gfn_risk": self.config.gfn.initial_risk, "liquidity": 0.72
        }
        self.kalman = KalmanFilter1D(self.config.kalman)
        self.bayesian = BayesianBetaUpdater(self.config.bayesian)
        self.gfn.reset(seed=self.config.random_seed)
        self.liquidity_model.reset(seed=self.config.random_seed)
        self.in_crisis_mode = False
        self._use_coupling = use_coupling
        self._coupling_strength = coupling_strength
        self.hyst.use_hysteresis = use_hysteresis
        self.hyst.crisis_kappa_multiplier = crisis_multiplier

        for _ in range(months):
            self.step(scenario, total_months=months)
        return pd.DataFrame(self.history)

    def get_network_for_viz(self):
        return self.population.get_network_for_viz()


def escalation_persistence(df: pd.DataFrame) -> Dict:
    if df is None or df.empty or "gfn_risk" not in df.columns:
        return {}
    peak_idx = df["gfn_risk"].idxmax()
    esc = df["escalation"]
    post = esc.loc[peak_idx:]
    esc_at = float(esc.loc[peak_idx])
    esc_max = float(post.max())
    lag = int(post.idxmax() - peak_idx)
    target = esc_at + 0.5 * (esc_max - esc_at)
    below = post[post <= target]
    half = int(below.index.min() - peak_idx) if len(below) else None
    out = {
        "gfn_peak_month": int(df.loc[peak_idx, "month"]),
        "gfn_peak_value": float(df.loc[peak_idx, "gfn_risk"]),
        "lag_months": lag,
        "half_life_months": half,
        "esc_max_after": esc_max,
    }
    if "in_crisis_mode" in df.columns:
        out["crisis_months_after_peak"] = int(df.loc[peak_idx:, "in_crisis_mode"].sum())
    return out


# =============================================================================
# RESEARCH OUTPUT PATHS
# =============================================================================

OUT_DIR = Path("research_output")
FIG_DIR = OUT_DIR / "figures"
DATA_DIR = FIG_DIR / "data"

PALETTE = {
    "primary": "#8B5CF6",
    "secondary": "#3B82F6",
    "accent": "#DC2626",
    "warning": "#F59E0B",
    "success": "#10B981",
    "neutral": "#64748B",
    "grid": "rgba(148,163,184,0.15)",
}

ExportFormat = Literal["png", "svg", "pdf", "eps", "webp"]


# =============================================================================
# RESEARCH: (1) SOBOL
# =============================================================================

def _run_model_for_sensitivity(params: Dict[str, float], seed: int = 42) -> float:
    sdm_updates, gfn_updates = {}, {}
    for k, v in params.items():
        if k.startswith("gfn_"):
            gfn_updates[k[4:]] = v
        else:
            sdm_updates[k] = v

    sdm_cfg = replace(DEFAULT_CONFIG.sdm, **sdm_updates)
    gfn_cfg = replace(DEFAULT_CONFIG.gfn, **gfn_updates)
    cfg = replace(DEFAULT_CONFIG, random_seed=seed,
                  sdm=sdm_cfg, gfn=gfn_cfg, use_network_abm=False)
    model = NRDCPI_GFN_v22(config=cfg)
    df = model.run_simulation(months=36, scenario="gfn_crisis")
    if df is None or df.empty:
        return np.nan
    return float(df["escalation"].max())


def global_sensitivity_analysis(n_samples: int = 128, seed: int = 42) -> pd.DataFrame:
    try:
        from scipy.stats import qmc
    except ImportError:
        raise RuntimeError("Требуется scipy. Установите: pip install scipy")

    param_space = {
        "escalation_polarization_weight": (0.20, 0.70),
        "escalation_tone_weight":         (0.10, 0.50),
        "escalation_sipri_weight":        (0.04, 0.25),
        "escalation_rand_weight":         (0.04, 0.25),
        "escalation_network_weight":      (0.05, 0.30),
        "escalation_trust_effect":        (0.10, 0.40),
        "escalation_innovation_effect":   (0.03, 0.18),
        "escalation_credit_effect":       (0.02, 0.12),
        "escalation_shock_effect":        (0.20, 0.80),
        "nrdcpi_escalation_weight":       (15.0, 45.0),
        "nrdcpi_trust_weight":            (4.0, 18.0),
        "nrdcpi_polarization_weight":     (8.0, 25.0),
        "nrdcpi_convergence_rate":        (0.05, 0.25),
        "trust_growth_rate":              (0.005, 0.040),
        "trust_escalation_penalty":       (0.02, 0.10),
        "polarization_escalation_effect": (0.003, 0.025),
        "polarization_ideology_effect":   (0.10, 0.60),
        "gfn_base_drift_mean":            (0.001, 0.008),
        "gfn_shock_peak":                 (0.015, 0.060),
        "gfn_contagion_coeff":            (0.003, 0.018),
        "gfn_volatility_std":             (0.003, 0.018),
        "gfn_coupling_strength":          (0.04, 0.30),
    }

    names = list(param_space.keys())
    d = len(names)
    bounds = np.array([param_space[n] for n in names])

    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    base = sampler.random(n_samples)
    A, B = base[:, :d], base[:, d:2 * d]

    def scale(x):
        return bounds[:, 0] + x * (bounds[:, 1] - bounds[:, 0])

    A_s, B_s = scale(A), scale(B)

    AB = np.empty((n_samples, d, d))
    for i in range(d):
        AB[:, i, :] = A_s.copy()
        AB[:, i, i] = B_s[:, i]

    print(f"[sensitivity] Прогон {n_samples * (d + 2)} симуляций...")
    y_A = np.array([_run_model_for_sensitivity(dict(zip(names, A_s[j])))
                    for j in range(n_samples)])
    y_B = np.array([_run_model_for_sensitivity(dict(zip(names, B_s[j])))
                    for j in range(n_samples)])
    y_AB = np.array([
        [_run_model_for_sensitivity(dict(zip(names, AB[j, i, :])))
         for i in range(d)]
        for j in range(n_samples)
    ])

    var_y = np.var(np.concatenate([y_A, y_B]))
    S1, ST = np.zeros(d), np.zeros(d)
    for i in range(d):
        S1[i] = np.mean(y_B * (y_AB[:, i] - y_A)) / var_y
        ST[i] = 0.5 * np.mean((y_A - y_AB[:, i]) ** 2) / var_y

    result = pd.DataFrame({
        "parameter": names,
        "S1_first_order": S1,
        "ST_total_order": ST,
        "ST_minus_S1_interaction": ST - S1,
    }).sort_values("ST_total_order", ascending=False).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_DIR / "sobol_sensitivity.csv", index=False)
    print(f"[sensitivity] Сохранено: {OUT_DIR / 'sobol_sensitivity.csv'}")
    return result


# =============================================================================
# RESEARCH: (2) КАЛИБРОВКА κ
# =============================================================================

def _historical_proxy_data() -> pd.DataFrame:
    episodes = [
        (2007, 3, 0.52, 0.42, "pre-GFC"), (2008, 4, 0.81, 0.58, "GFC peak"),
        (2009, 2, 0.71, 0.55, "GFC aftermath"), (2010, 1, 0.58, 0.47, "Eurozone stress"),
        (2011, 3, 0.55, 0.51, "Arab Spring"), (2014, 1, 0.48, 0.62, "Crimea"),
        (2015, 2, 0.46, 0.58, "Donbas"), (2016, 4, 0.44, 0.52, "Syria"),
        (2018, 2, 0.49, 0.55, "Trade war onset"), (2019, 4, 0.51, 0.57, "Trade war peak"),
        (2020, 1, 0.78, 0.63, "COVID shock"), (2020, 4, 0.62, 0.60, "COVID recovery"),
        (2021, 4, 0.56, 0.66, "pre-Ukraine"), (2022, 1, 0.68, 0.85, "Ukraine invasion"),
        (2022, 3, 0.72, 0.88, "Ukraine stalemate"), (2023, 2, 0.61, 0.81, "Counteroffensive"),
        (2024, 1, 0.58, 0.79, "Attrition"), (2025, 1, 0.55, 0.77, "Frozen"),
        (2026, 1, 0.57, 0.80, "Continued"),
    ]
    df = pd.DataFrame(episodes,
                      columns=["year", "quarter", "gfn_risk", "escalation", "note"])
    df["t"] = df["year"] + (df["quarter"] - 1) / 4.0
    return df


def calibrate_coupling_kappa(threshold_act: float = 0.45,
                              n_grid: int = 200, seed: int = 42) -> Dict[str, Any]:
    df = _historical_proxy_data()
    excess = np.maximum(0.0, df["gfn_risk"].values - threshold_act)
    esc_obs = df["escalation"].values
    kappas = np.linspace(0.0, 1.0, n_grid)
    mse = np.array([np.mean((k * excess - (esc_obs - esc_obs.min())) ** 2)
                    for k in kappas])
    kappa_opt = float(kappas[np.argmin(mse)])

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(500):
        idx = rng.integers(0, len(df), len(df))
        ex_b = excess[idx]
        esc_b = esc_obs[idx] - esc_obs.min()
        mse_b = np.array([np.mean((k * ex_b - esc_b) ** 2) for k in kappas])
        boot.append(kappas[np.argmin(mse_b)])
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])

    result = {
        "kappa_opt": kappa_opt,
        "kappa_ci_95": [float(ci_low), float(ci_high)],
        "threshold_act": threshold_act,
        "n_episodes": len(df),
        "mse_min": float(mse.min()),
        "grid_resolution": n_grid,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"kappa": kappas, "mse": mse}).to_csv(
        OUT_DIR / "kappa_calibration_curve.csv", index=False)
    with open(OUT_DIR / "kappa_calibration.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"[calibration] κ_opt = {kappa_opt:.4f} "
          f"(95% CI: {ci_low:.4f}–{ci_high:.4f})")
    return result


# =============================================================================
# RESEARCH: (3) ТОПОЛОГИИ
# =============================================================================

def _build_network(topology: str, n: int, seed: int = 42) -> nx.Graph:
    if topology == "ws":
        return nx.watts_strogatz_graph(n, 8, 0.12, seed=seed)
    if topology == "ba":
        return nx.barabasi_albert_graph(n, 4, seed=seed)
    if topology == "core_periphery":
        n_core = max(2, int(0.2 * n))
        G = nx.Graph()
        G.add_nodes_from(range(n))
        core = list(range(n_core))
        peri = list(range(n_core, n))
        for i in core:
            for j in core:
                if i < j:
                    G.add_edge(i, j)
        rng = np.random.default_rng(seed)
        for p in peri:
            for t in rng.choice(core, size=min(2, n_core), replace=False):
                G.add_edge(p, int(t))
        return G
    raise ValueError(f"Unknown topology: {topology}")


def compare_topologies(topologies: List[str] = ("ws", "ba", "core_periphery"),
                       months: int = 36, scenario: str = "gfn_crisis",
                       n_seeds: int = 10) -> pd.DataFrame:
    rows = []
    for topo in topologies:
        for s in range(n_seeds):
            seed = 42 + s
            cfg = replace(DEFAULT_CONFIG, random_seed=seed, use_network_abm=True)
            model = NRDCPI_GFN_v22(config=cfg)
            model.population.network = _build_network(
                topo, len(model.population.agents), seed=seed)
            df = model.run_simulation(months=months, scenario=scenario)
            pers = escalation_persistence(df)
            rows.append({
                "topology": topo, "seed": seed,
                "esc_max": float(df["escalation"].max()),
                "esc_final": float(df["escalation"].iloc[-1]),
                "lag_months": pers.get("lag_months"),
                "half_life_months": pers.get("half_life_months"),
                "crisis_months_after_peak": pers.get("crisis_months_after_peak"),
                "network_density": model.population.network_stats.get("network_density"),
                "polarization_proxy": model.population.network_stats.get("polarization_proxy"),
            })
        print(f"[topology] {topo}: обработано {n_seeds} seed'ов")

    df_res = pd.DataFrame(rows)
    summary = df_res.groupby("topology").agg(
        esc_max_mean=("esc_max", "mean"),
        esc_max_std=("esc_max", "std"),
        lag_mean=("lag_months", "mean"),
        lag_std=("lag_months", "std"),
        half_life_mean=("half_life_months", "mean"),
        crisis_months_mean=("crisis_months_after_peak", "mean"),
        density_mean=("network_density", "mean"),
        polarization_mean=("polarization_proxy", "mean"),
    ).reset_index()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(OUT_DIR / "topology_comparison_raw.csv", index=False)
    summary.to_csv(OUT_DIR / "topology_comparison_summary.csv", index=False)
    print(f"[topology] Сохранено: {OUT_DIR / 'topology_comparison_summary.csv'}")
    return summary


# =============================================================================
# RESEARCH: (4) MONTE CARLO
# =============================================================================

def monte_carlo_seeds(n_seeds: int = 30, months: int = 36,
                      scenario: str = "gfn_crisis") -> pd.DataFrame:
    rows = []
    for s in range(n_seeds):
        seed = 1000 + s
        cfg = replace(DEFAULT_CONFIG, random_seed=seed)
        model = NRDCPI_GFN_v22(config=cfg)
        df = model.run_simulation(months=months, scenario=scenario)
        pers = escalation_persistence(df)
        rows.append({
            "seed": seed,
            "esc_max": float(df["escalation"].max()),
            "esc_final": float(df["escalation"].iloc[-1]),
            "nrdcpi_final": float(df["nrdcpi_filtered"].iloc[-1]),
            "gfn_peak": float(df["gfn_risk"].max()),
            "lag_months": pers.get("lag_months"),
            "half_life_months": pers.get("half_life_months"),
            "crisis_months_after_peak": pers.get("crisis_months_after_peak"),
            "in_crisis_total": int(df["in_crisis_mode"].sum()),
        })
        if (s + 1) % 10 == 0:
            print(f"[monte_carlo] {s + 1}/{n_seeds} seed'ов обработано")

    df_res = pd.DataFrame(rows)
    stats = df_res.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(OUT_DIR / "monte_carlo_raw.csv", index=False)
    stats.to_csv(OUT_DIR / "monte_carlo_stats.csv")
    print(f"[monte_carlo] Сохранено: {OUT_DIR / 'monte_carlo_stats.csv'}")
    return stats


# =============================================================================
# RESEARCH: (5) ЮНИТ-ТЕСТЫ
# =============================================================================

def _test_clipping_invariants() -> List[Tuple[str, bool, str]]:
    cfg = replace(DEFAULT_CONFIG, random_seed=7)
    model = NRDCPI_GFN_v22(config=cfg)
    df = model.run_simulation(months=48, scenario="nuclear")
    checks = {
        "territory_in_range": df["territory"].between(12.0, 88.0).all(),
        "escalation_in_range": df["escalation"].between(0.02, 0.98).all(),
        "nrdcpi_in_range": df["nrdcpi"].between(14.0, 96.0).all(),
        "trust_in_range": df["trust"].between(0.08, 0.92).all(),
        "polarization_in_range": df["polarization"].between(0.02, 0.92).all(),
        "pro_axis_in_range": df["pro_axis_ratio"].between(0.04, 0.96).all(),
        "liquidity_in_range": df["liquidity"].between(0.15, 0.95).all(),
        "gfn_risk_in_range": df["gfn_risk"].between(0.05, 0.98).all(),
    }
    return [(n, bool(ok), "" if ok else "нарушение границ")
            for n, ok in checks.items()]


def _test_abm_mass_conservation() -> List[Tuple[str, bool, str]]:
    cfg = replace(DEFAULT_CONFIG, random_seed=11)
    model = NRDCPI_GFN_v22(config=cfg)
    n_before = len(model.population.agents)
    e_before = model.population.network.number_of_edges()
    for _ in range(12):
        model.step(scenario="base", total_months=12)
    n_after = len(model.population.agents)
    e_after = model.population.network.number_of_edges()
    return [
        ("abm_agent_count_conserved", n_before == n_after,
         f"{n_before} → {n_after}"),
        ("abm_network_edges_conserved", e_before == e_after,
         f"{e_before} → {e_after}"),
    ]


def _test_rk4_convergence() -> List[Tuple[str, bool, str]]:
    base_state = np.array([49.0, 0.41, 46.5, 0.44, 0.34, 0.46])
    params = {
        "external_shock": 0.01, "gdelt_tone": -0.30, "abm_factors": {},
        "external": {"ucdp_intensity": 0.41, "sipri_ratio": 1.18,
                     "vdem_libdem_avg": 0.39, "rand_strategic": 0.49},
        "trust_break_factor": 1.0, "gfn_risk": 0.60, "gfn_coupling": 0.12,
    }
    ref = run_sdm_step(base_state.copy(), params,
                        replace(DEFAULT_CONFIG.sdm, dt=0.001), months=1.0)
    errors = []
    for dt in [0.05, 0.025, 0.0125]:
        out = run_sdm_step(base_state.copy(), params,
                            replace(DEFAULT_CONFIG.sdm, dt=dt), months=1.0)
        errors.append((dt, float(np.linalg.norm(out - ref))))
    results = [(
        "rk4_error_monotone_decreasing",
        all(errors[i][1] >= errors[i + 1][1] for i in range(len(errors) - 1)),
        "; ".join(f"dt={dt}: err={err:.6f}" for dt, err in errors),
    )]
    if len(errors) >= 2:
        ratio = errors[0][1] / max(errors[1][1], 1e-12)
        results.append(("rk4_order_plausible", ratio > 2.0,
                        f"ratio(dt=0.05/dt=0.025) = {ratio:.2f}"))
    return results


def _test_kalman_invariants() -> List[Tuple[str, bool, str]]:
    kf = KalmanFilter1D(DEFAULT_CONFIG.kalman)
    o1 = kf.update(50.0)
    o2 = kf.update(50.0)
    return [
        ("kalman_gain_decreases_on_repeat",
         o2["kalman_gain"] <= o1["kalman_gain"],
         f"K1={o1['kalman_gain']:.4f}, K2={o2['kalman_gain']:.4f}"),
        ("kalman_variance_decreases",
         o2["variance"] <= o1["variance"],
         f"P1={o1['variance']:.4f}, P2={o2['variance']:.4f}"),
        ("kalman_gain_in_01", 0.0 <= o1["kalman_gain"] <= 1.0,
         f"K={o1['kalman_gain']:.4f}"),
    ]


def _test_gfn_invariants() -> List[Tuple[str, bool, str]]:
    gfn = GFNRiskModel(DEFAULT_CONFIG.gfn, seed=42)
    history = [gfn.current_risk]
    for _ in range(120):
        history.append(gfn.step(total_months=60, scenario_multiplier=2.2))
    arr = np.array(history)
    return [
        ("gfn_history_in_range",
         bool(np.all((arr >= 0.05) & (arr <= 0.98))),
         f"min={arr.min():.4f}, max={arr.max():.4f}"),
        ("gfn_history_length_correct",
         len(gfn.history) == 121, f"len={len(gfn.history)}"),
    ]


def run_unit_tests() -> pd.DataFrame:
    all_results: List[Tuple[str, str, bool, str]] = []
    suites = [
        ("clipping", _test_clipping_invariants),
        ("abm_mass", _test_abm_mass_conservation),
        ("rk4", _test_rk4_convergence),
        ("kalman", _test_kalman_invariants),
        ("gfn", _test_gfn_invariants),
    ]
    for suite_name, fn in suites:
        try:
            for name, ok, note in fn():
                all_results.append((suite_name, name, ok, note))
        except Exception as e:
            all_results.append((suite_name, "EXCEPTION", False, str(e)))

    df = pd.DataFrame(all_results, columns=["suite", "test", "passed", "note"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "unit_tests.csv", index=False)
    n_pass = int(df["passed"].sum())
    print(f"\n[unit_tests] {n_pass}/{len(df)} тестов пройдено")
    if n_pass < len(df):
        print(df[~df["passed"]].to_string(index=False))
    return df


# =============================================================================
# VISUALIZATION: SOBOL / KAPPA / TOPOLOGY / MONTE CARLO
# =============================================================================

def plot_sobol_bar(csv_path: Optional[Path] = None, top_n: int = 15) -> go.Figure:
    csv_path = csv_path or (OUT_DIR / "sobol_sensitivity.csv")
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Не найден {csv_path}. "
                                 f"Запустите с --only sensitivity")
    df = pd.read_csv(csv_path).head(top_n).iloc[::-1]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=df["parameter"], x=df["ST_total_order"],
                         name="ST (total order)", orientation="h",
                         marker=dict(color=PALETTE["secondary"], opacity=0.55)))
    fig.add_trace(go.Bar(y=df["parameter"], x=df["S1_first_order"],
                         name="S1 (first order)", orientation="h",
                         marker=dict(color=PALETTE["primary"])))
    fig.add_trace(go.Scatter(y=df["parameter"], x=df["ST_minus_S1_interaction"],
                             mode="markers", name="ST − S1 (interactions)",
                             marker=dict(color=PALETTE["accent"], size=10,
                                         symbol="diamond",
                                         line=dict(width=1, color="white"))))
    fig.update_layout(
        title=dict(text="<b>Глобальный анализ чувствительности (Sobol)</b>"
                        "<br><sub>Индексы первого и полного порядка</sub>",
                   font=dict(size=16, color="white")),
        barmode="overlay", height=max(420, 26 * len(df) + 120),
        margin=dict(l=220, r=40, t=90, b=50),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=12),
        xaxis=dict(title="Индекс Sobol", gridcolor=PALETTE["grid"]),
        yaxis=dict(gridcolor=PALETTE["grid"]),
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
    )
    return fig


def plot_kappa_calibration(curve_path: Optional[Path] = None,
                            json_path: Optional[Path] = None) -> go.Figure:
    curve_path = curve_path or (OUT_DIR / "kappa_calibration_curve.csv")
    json_path = json_path or (OUT_DIR / "kappa_calibration.json")
    if not Path(curve_path).exists() or not Path(json_path).exists():
        raise FileNotFoundError("Нет данных калибровки κ. "
                                 "Запустите с --only calibration")
    curve = pd.read_csv(curve_path)
    with open(json_path) as f:
        meta = json.load(f)

    fig = go.Figure()
    fig.add_vrect(x0=meta["kappa_ci_95"][0], x1=meta["kappa_ci_95"][1],
                  fillcolor="rgba(139,92,246,0.18)", line_width=0,
                  annotation_text=f"95% CI: [{meta['kappa_ci_95'][0]:.3f}, "
                                  f"{meta['kappa_ci_95'][1]:.3f}]",
                  annotation_position="top left",
                  annotation_font=dict(color="white", size=11))
    fig.add_trace(go.Scatter(x=curve["kappa"], y=curve["mse"],
                             mode="lines", name="MSE(κ)",
                             line=dict(color=PALETTE["secondary"], width=2.5),
                             fill="tozeroy",
                             fillcolor="rgba(59,130,246,0.10)"))
    fig.add_trace(go.Scatter(x=[meta["kappa_opt"]], y=[meta["mse_min"]],
                             mode="markers+text",
                             name=f"κ_opt = {meta['kappa_opt']:.3f}",
                             marker=dict(color=PALETTE["accent"], size=14,
                                         symbol="star",
                                         line=dict(width=1.5, color="white")),
                             text=[f"κ_opt = {meta['kappa_opt']:.3f}"],
                             textposition="top right",
                             textfont=dict(color="white", size=12)))
    fig.update_layout(
        title=dict(text="<b>Калибровка coupling κ</b>"
                        f"<br><sub>θ_act = {meta['threshold_act']}, "
                        f"N = {meta['n_episodes']}, "
                        f"сетка = {meta['grid_resolution']}</sub>",
                   font=dict(size=16, color="white")),
        height=460, margin=dict(l=80, r=40, t=100, b=60),
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=12),
        xaxis=dict(title="κ (coupling strength)", gridcolor=PALETTE["grid"]),
        yaxis=dict(title="MSE", gridcolor=PALETTE["grid"]),
        legend=dict(orientation="h", y=1.06, x=0.5, xanchor="center"),
    )
    return fig


def plot_topology_boxplots(raw_path: Optional[Path] = None,
                            metrics: Optional[list] = None) -> go.Figure:
    raw_path = raw_path or (OUT_DIR / "topology_comparison_raw.csv")
    if not Path(raw_path).exists():
        raise FileNotFoundError("Нет данных топологий. "
                                 "Запустите с --only topology")
    df = pd.read_csv(raw_path)
    metrics = metrics or ["esc_max", "esc_final", "lag_months",
                          "half_life_months", "crisis_months_after_peak",
                          "polarization_proxy"]
    topo_labels = {"ws": "Watts–Strogatz", "ba": "Barabási–Albert",
                   "core_periphery": "Core–periphery"}
    topo_colors = {"ws": PALETTE["secondary"], "ba": PALETTE["warning"],
                   "core_periphery": PALETTE["accent"]}
    n_cols = 3
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[m.replace("_", " ") for m in metrics],
                        vertical_spacing=0.14, horizontal_spacing=0.10)
    legend_shown = set()
    for idx, metric in enumerate(metrics):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        for topo in df["topology"].unique():
            sub = df[df["topology"] == topo][metric].dropna()
            if len(sub) == 0:
                continue
            show = topo not in legend_shown
            fig.add_trace(go.Box(
                y=sub, name=topo_labels.get(topo, topo),
                marker=dict(color=topo_colors.get(topo, PALETTE["neutral"])),
                boxmean="sd", legendgroup=topo, showlegend=show,
            ), row=r, col=c)
            legend_shown.add(topo)
    fig.update_layout(
        title=dict(text="<b>Устойчивость выводов к топологии сети</b>"
                        "<br><sub>Boxplot'ы метрик по трём топологиям</sub>",
                   font=dict(size=16, color="white")),
        height=320 * n_rows + 120, margin=dict(l=60, r=40, t=110, b=60),
        template="plotly_dark", boxmode="group",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
        legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"])
    return fig


def plot_monte_carlo_distribution(raw_path: Optional[Path] = None,
                                    metrics: Optional[list] = None) -> go.Figure:
    raw_path = raw_path or (OUT_DIR / "monte_carlo_raw.csv")
    if not Path(raw_path).exists():
        raise FileNotFoundError("Нет данных Monte Carlo. "
                                 "Запустите с --only monte_carlo")
    df = pd.read_csv(raw_path)
    metrics = metrics or ["esc_max", "esc_final", "lag_months",
                          "half_life_months", "crisis_months_after_peak",
                          "in_crisis_total"]
    n_cols = 3
    n_rows = int(np.ceil(len(metrics) / n_cols))
    fig = make_subplots(rows=n_rows, cols=n_cols,
                        subplot_titles=[m.replace("_", " ") for m in metrics],
                        vertical_spacing=0.16, horizontal_spacing=0.10)
    for idx, metric in enumerate(metrics):
        r, c = idx // n_cols + 1, idx % n_cols + 1
        vals = df[metric].dropna().values
        if len(vals) == 0:
            continue
        p5, p50, p95 = np.percentile(vals, [5, 50, 95])
        fig.add_trace(go.Histogram(
            x=vals, nbinsx=min(20, max(8, len(vals) // 3)),
            marker=dict(color=PALETTE["secondary"],
                        line=dict(color="rgba(255,255,255,0.3)", width=0.5)),
            opacity=0.85, showlegend=False,
        ), row=r, col=c)
        for p, color, label in [(p5, PALETTE["success"], "p5"),
                                 (p50, PALETTE["primary"], "p50"),
                                 (p95, PALETTE["accent"], "p95")]:
            fig.add_vline(x=p, line_dash="dash", line_color=color,
                          line_width=1.5,
                          annotation_text=f"{label}={p:.3f}",
                          annotation_position="top",
                          annotation_font=dict(color=color, size=10),
                          row=r, col=c)
    fig.update_layout(
        title=dict(text="<b>Распределения метрик по seed'ам (Monte Carlo)</b>"
                        "<br><sub>Перцентили 5%, 50%, 95%</sub>",
                   font=dict(size=16, color="white")),
        height=320 * n_rows + 120, margin=dict(l=60, r=40, t=110, b=60),
        template="plotly_dark", bargap=0.05,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white", size=12),
    )
    fig.update_xaxes(gridcolor=PALETTE["grid"])
    fig.update_yaxes(gridcolor=PALETTE["grid"])
    return fig


# =============================================================================
# DASHBOARD
# =============================================================================

def build_dashboard_html(out_path: str = "dashboard.html") -> str:
    from plotly.io import to_html
    blocks, warnings = [], []

    def _try(fn, title):
        try:
            fig = fn()
            blocks.append(to_html(
                fig, full_html=False,
                include_plotlyjs=("cdn" if not blocks else False),
                config={"displayModeBar": True, "responsive": True}))
        except Exception as e:
            warnings.append(f"[{title}] {e}")

    _try(plot_sobol_bar, "Sobol")
    _try(plot_kappa_calibration, "Kappa")
    _try(plot_topology_boxplots, "Topology")
    _try(plot_monte_carlo_distribution, "MonteCarlo")

    if not blocks:
        raise RuntimeError("Нет данных для дашборда. "
                           "Сначала запустите пайплайн.")

    html = """<!DOCTYPE html><html lang="ru"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>NRDCPI_GFN_v23 — Research Dashboard</title>
<style>
  body{background:#0F172A;color:#F1F5F9;
       font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       margin:0;padding:0 24px 48px 24px;}
  header{padding:32px 0 16px 0;
         border-bottom:1px solid rgba(148,163,184,0.15);margin-bottom:24px;}
  h1{font-size:1.8rem;font-weight:700;margin:0 0 6px 0;
     background:linear-gradient(90deg,#8B5CF6,#3B82F6,#DC2626);
     -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .subtitle{color:#94A3B8;font-size:0.95rem;}
  .panel{background:#1E293B;border-radius:12px;padding:16px;
         margin-bottom:24px;box-shadow:0 1px 3px rgba(0,0,0,0.3);}
  .warnings{background:rgba(245,158,11,0.08);border-left:3px solid #F59E0B;
            padding:12px 16px;margin-bottom:24px;color:#FCD34D;
            font-size:0.9rem;border-radius:6px;}
</style></head><body>
<header><h1>NRDCPI_GFN_v23 — Research Dashboard</h1>
<div class="subtitle">Sobol · κ calibration · topology · Monte Carlo</div>
</header>"""
    if warnings:
        html += '<div class="warnings"><b>Предупреждения:</b><br>'
        html += "<br>".join(f"• {w}" for w in warnings) + "</div>"
    for block in blocks:
        html += f'<div class="panel">{block}</div>'
    html += "</body></html>"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[dashboard] Сохранено: {out_path}")
    return out_path


# =============================================================================
# EXPORT PNG/SVG/PDF
# =============================================================================

def _check_kaleido() -> None:
    try:
        import kaleido  # noqa: F401
    except ImportError as e:
        raise ImportError("Требуется kaleido. Установите: pip install kaleido\n"
                          f"Ошибка: {e}")


def export_static(fig: go.Figure, path: Union[str, Path],
                  width: int = 1400, height: int = 800,
                  scale: float = 2.0,
                  format: ExportFormat = "png") -> Path:
    _check_kaleido()
    path = Path(path)
    if path.suffix.lower().lstrip(".") != format:
        path = path.with_suffix(f".{format}")
    fig_static = go.Figure(fig)
    fig_static.update_layout(hovermode=False)
    fig_static.write_image(str(path), width=width, height=height,
                            scale=scale, format=format)
    print(f"[export_static] {path} ({format.upper()}, "
          f"{width}×{height} × {scale})")
    return path


def export_all_formats(fig: go.Figure, base_path: Union[str, Path],
                       width: int = 1400, height: int = 800,
                       scale: float = 2.0,
                       formats: tuple = ("png", "svg", "pdf")) -> dict:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    return {fmt: export_static(fig, base_path, width=width, height=height,
                                scale=scale, format=fmt)
            for fmt in formats}


def export_journal_set(out_dir: Union[str, Path] = FIG_DIR,
                       journal_preset: str = "double_column") -> dict:
    sizes = {"single_column": (800, 600), "one_and_half": (1200, 800),
             "double_column": (1800, 900), "presentation": (1920, 1080)}
    width, height = sizes[journal_preset]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = {}
    for name, fn in [("fig1_sobol", plot_sobol_bar),
                     ("fig2_kappa", plot_kappa_calibration),
                     ("fig3_topology", plot_topology_boxplots),
                     ("fig4_monte_carlo", plot_monte_carlo_distribution)]:
        try:
            figures[name] = fn()
        except Exception as e:
            print(f"[export_journal_set] пропущено {name}: {e}")
    if not figures:
        raise RuntimeError("Нет данных для экспорта.")

    all_paths = {}
    for name, fig in figures.items():
        all_paths[name] = export_all_formats(
            fig, out_dir / name, width=width, height=height,
            scale=2.0, formats=("png", "svg", "pdf"))
    print(f"\n[export_journal_set] Готово: {out_dir} "
          f"({journal_preset}, {width}×{height})")
    return all_paths


def export_figure_data_to_xlsx(out_dir: Union[str, Path] = DATA_DIR) -> dict:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        raise ImportError("Требуется openpyxl. Установите: pip install openpyxl")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    sobol_csv = OUT_DIR / "sobol_sensitivity.csv"
    if sobol_csv.exists():
        p = out_dir / "fig1_sobol_data.xlsx"
        pd.read_csv(sobol_csv).to_excel(p, index=False, sheet_name="Sobol_indices")
        results["fig1_sobol"] = p

    kappa_curve = OUT_DIR / "kappa_calibration_curve.csv"
    kappa_json = OUT_DIR / "kappa_calibration.json"
    if kappa_curve.exists() and kappa_json.exists():
        with open(kappa_json) as f:
            meta = json.load(f)
        p = out_dir / "fig2_kappa_data.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            pd.read_csv(kappa_curve).to_excel(w, index=False, sheet_name="MSE_curve")
            pd.DataFrame([meta]).to_excel(w, index=False, sheet_name="Calibration_meta")
        results["fig2_kappa"] = p

    topo_raw = OUT_DIR / "topology_comparison_raw.csv"
    topo_sum = OUT_DIR / "topology_comparison_summary.csv"
    if topo_raw.exists():
        p = out_dir / "fig3_topology_data.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            pd.read_csv(topo_raw).to_excel(w, index=False, sheet_name="Raw")
            if topo_sum.exists():
                pd.read_csv(topo_sum).to_excel(w, index=False, sheet_name="Summary")
        results["fig3_topology"] = p

    mc_raw = OUT_DIR / "monte_carlo_raw.csv"
    mc_stats = OUT_DIR / "monte_carlo_stats.csv"
    if mc_raw.exists():
        p = out_dir / "fig4_monte_carlo_data.xlsx"
        with pd.ExcelWriter(p, engine="openpyxl") as w:
            pd.read_csv(mc_raw).to_excel(w, index=False, sheet_name="Raw")
            if mc_stats.exists():
                pd.read_csv(mc_stats).to_excel(w, index=False, sheet_name="Stats")
        results["fig4_monte_carlo"] = p

    print(f"\n[export_xlsx] Сохранено: {out_dir}")
    for name, p in results.items():
        print(f"  {name}: {p.name}")
    return results


# =============================================================================
# CLI ORCHESTRATOR
# =============================================================================

def _cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        description="NRDCPI_GFN_v23: модель + исследовательский пайплайн")
    parser.add_argument("--only", type=str, default=None,
                        choices=["sensitivity", "calibration", "topology",
                                 "monte_carlo", "unit_tests",
                                 "dashboard", "export", "xlsx"])
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--n_seeds", type=int, default=None)
    parser.add_argument("--preset", type=str, default="double_column",
                        choices=["single_column", "one_and_half",
                                 "double_column", "presentation"])
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    for d in (OUT_DIR, FIG_DIR, DATA_DIR):
        d.mkdir(parents=True, exist_ok=True)

    n_samples = args.n_samples if args.n_samples else (32 if args.fast else 128)
    n_seeds = args.n_seeds if args.n_seeds else (10 if args.fast else 30)

    if args.only == "sensitivity":
        global_sensitivity_analysis(n_samples=n_samples); return
    if args.only == "calibration":
        calibrate_coupling_kappa(); return
    if args.only == "topology":
        compare_topologies(n_seeds=n_seeds); return
    if args.only == "monte_carlo":
        monte_carlo_seeds(n_seeds=n_seeds); return
    if args.only == "unit_tests":
        run_unit_tests(); return
    if args.only == "dashboard":
        build_dashboard_html()
        if args.open:
            webbrowser.open(f"file://{Path('dashboard.html').resolve()}")
        return
    if args.only == "export":
        export_journal_set(journal_preset=args.preset); return
    if args.only == "xlsx":
        export_figure_data_to_xlsx(); return

    t0 = time.time()
    print("=" * 72)
    print(f"  ПОЛНЫЙ ПАЙПЛАЙН  (N_samples={n_samples}, N_seeds={n_seeds})")
    print("=" * 72)

    print("\n[1/8] ЮНИТ-ТЕСТЫ"); run_unit_tests()
    print("\n[2/8] КАЛИБРОВКА κ"); calibrate_coupling_kappa()
    print("\n[3/8] SOBOL"); global_sensitivity_analysis(n_samples=n_samples)
    print("\n[4/8] ТОПОЛОГИИ"); compare_topologies(n_seeds=n_seeds)
    print("\n[5/8] MONTE CARLO"); monte_carlo_seeds(n_seeds=n_seeds)
    print("\n[6/8] ЭКСПОРТ PNG/SVG/PDF")
    try:
        export_journal_set(journal_preset=args.preset)
    except Exception as e:
        print(f"[export] пропущено: {e}")
    print("\n[7/8] ЭКСПОРТ .XLSX")
    try:
        export_figure_data_to_xlsx()
    except Exception as e:
        print(f"[xlsx] пропущено: {e}")
    print("\n[8/8] DASHBOARD")
    try:
        build_dashboard_html()
    except Exception as e:
        print(f"[dashboard] пропущено: {e}")

    print(f"\nОбщее время: {time.time() - t0:.1f}s")
    if args.open and Path("dashboard.html").exists():
        webbrowser.open(f"file://{Path('dashboard.html').resolve()}")


# =============================================================================
# STREAMLIT APP
# =============================================================================

def _streamlit_main():
    import streamlit as st

    st.set_page_config(page_title="NRDCPI + GFN v23", layout="wide",
                       page_icon="🕷️", initial_sidebar_state="expanded")
    st.markdown("""
        <style>
        .stApp { background-color: #0F172A; }
        .main-header {
            font-size: 2rem; font-weight: 700; padding: 0.5rem 0;
            background: linear-gradient(90deg, #8B5CF6, #3B82F6, #DC2626);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        </style>
    """, unsafe_allow_html=True)
    st.markdown('<div class="main-header">NRDCPI + GFN v23</div>',
                unsafe_allow_html=True)
    st.caption("GFN risk → coupling / hysteresis / liquidity → peripheral conflict")

    if "model" not in st.session_state:
        st.session_state.model = NRDCPI_GFN_v22(config=DEFAULT_CONFIG)
    model = st.session_state.model

    with st.sidebar:
        st.markdown("### Controls")
        scenario = st.selectbox(
            "Scenario",
            ["base", "peace", "war", "freeze", "nuclear", "gfn_crisis"],
            format_func=lambda x: {
                "base": "Baseline", "peace": "Peace", "war": "War",
                "freeze": "Freeze", "nuclear": "Nuclear",
                "gfn_crisis": "GFN Crisis"}[x])
        months = st.slider("Horizon (months)", 12, 60, 36)

        st.markdown("---")
        st.markdown("### Coupling")
        use_coupling = st.checkbox("Enable GFN coupling", value=True)
        coupling = st.slider("Coupling strength", 0.0, 0.30, 0.12, 0.01)

        st.markdown("---")
        st.markdown("### Hysteresis")
        use_hyst = st.checkbox("Enable crisis hysteresis", value=True)
        mult = st.slider("Crisis κ multiplier", 1.0, 4.0, 2.5, 0.1)

        if st.button("▶️ Run Simulation", type="primary",
                     use_container_width=True):
            with st.spinner(f"Running {months}-month simulation..."):
                results = model.run_simulation(
                    months, scenario,
                    use_coupling=use_coupling,
                    coupling_strength=coupling,
                    use_hysteresis=use_hyst,
                    crisis_multiplier=mult)
                st.session_state.results = results
                st.session_state.scenario = scenario
                st.rerun()

    if "results" in st.session_state and st.session_state.results is not None:
        df = st.session_state.results
        scen = st.session_state.scenario

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("NRDCPI", f"{df['nrdcpi_filtered'].iloc[-1]:.1f}")
        c2.metric("Escalation", f"{df['escalation'].iloc[-1]:.1%}")
        c3.metric("GFN Risk", f"{df['gfn_risk'].iloc[-1]:.3f}")
        c4.metric("Liquidity",
                  f"{df['liquidity'].iloc[-1]:.3f}" if "liquidity" in df.columns else "—")
        c5.metric("Polarization", f"{df['polarization'].iloc[-1]:.1%}")
        c6.metric("Crisis months",
                  int(df["in_crisis_mode"].sum()) if "in_crisis_mode" in df.columns else 0)

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
            "🕷️ GFN", "📉 Hysteresis", "💧 Liquidity",
            "🕸️ Financial Net", "📊 Escalation", "📈 NRDCPI / ABM",
            "🔬 Research"])

        with tab1:
            _render_gfn_panel(df)
        with tab2:
            _render_hysteresis_panel(df)
            pers = escalation_persistence(df)
            if pers:
                st.markdown("**Persistence after GFN peak**")
                st.json(pers)
        with tab3:
            _render_liquidity_panel(df)
        with tab4:
            _render_financial_network(float(df["gfn_risk"].iloc[-1]))
        with tab5:
            _render_escalation_barometer(df, scen)
        with tab6:
            _render_nrdcpi_dynamics(df)
            _render_network_viz(model)
        with tab7:
            st.subheader("Research pipeline")
            st.markdown("Запустите в терминале:")
            st.code(
                "python NRDCPI_GFN_v23.py --only unit_tests\n"
                "python NRDCPI_GFN_v23.py --only calibration\n"
                "python NRDCPI_GFN_v23.py --only sensitivity\n"
                "python NRDCPI_GFN_v23.py --only topology\n"
                "python NRDCPI_GFN_v23.py --only monte_carlo\n"
                "python NRDCPI_GFN_v23.py --only dashboard\n"
                "python NRDCPI_GFN_v23.py --only export\n"
                "python NRDCPI_GFN_v23.py --only xlsx",
                language="bash")

            for title, fn in [("Sobol sensitivity", plot_sobol_bar),
                              ("κ calibration", plot_kappa_calibration),
                              ("Topology comparison", plot_topology_boxplots),
                              ("Monte Carlo distributions",
                               plot_monte_carlo_distribution)]:
                st.markdown(f"### {title}")
                try:
                    st.plotly_chart(fn(), use_container_width=True)
                except FileNotFoundError as e:
                    st.info(f"{e}")

        with st.expander("Data (last 15 rows)"):
            st.dataframe(df.tail(15), use_container_width=True)
            st.download_button("Download CSV", df.to_csv(index=False),
                               file_name=f"nrdcpi_gfn_v23_{scen}.csv",
                               mime="text/csv")
    else:
        st.info("Выберите сценарий (рекомендуется **GFN Crisis**) "
                "и нажмите **Run Simulation**")


# ---------- Streamlit helper panels ----------

def _render_gfn_panel(df, threshold=0.68):
    import streamlit as st
    st.subheader("🕷️ GFN Systemic Risk")
    if df is None or df.empty:
        return
    months, gfn = df["month"], df["gfn_risk"]
    current = float(gfn.iloc[-1])
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number+delta", value=current,
        delta={"reference": 0.48, "increasing": {"color": "#DC2626"},
               "decreasing": {"color": "#10B981"}},
        title={"text": "GFN Risk", "font": {"size": 14}},
        gauge={
            "axis": {"range": [0, 1], "tickvals": [0, 0.45, 0.68, 1]},
            "bar": {"color": "#1E293B", "thickness": 0.25},
            "steps": [
                {"range": [0, 0.45], "color": "rgba(16,185,129,0.35)"},
                {"range": [0.45, 0.68], "color": "rgba(251,191,36,0.35)"},
                {"range": [0.68, 1], "color": "rgba(220,38,38,0.4)"},
            ],
            "threshold": {"line": {"color": "#7F1D1D", "width": 3},
                          "value": threshold}
        }))
    fig_g.update_layout(height=260, margin=dict(t=50, b=10, l=10, r=10),
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"))
    st.plotly_chart(fig_g, use_container_width=True)

    fig_t = go.Figure()
    fig_t.add_hrect(y0=0.68, y1=1.0,
                    fillcolor="rgba(220,38,38,0.15)", line_width=0)
    fig_t.add_trace(go.Scatter(x=months, y=gfn, mode="lines+markers",
                               line=dict(color="#8B5CF6", width=2.5),
                               name="GFN Risk", fill="tozeroy",
                               fillcolor="rgba(139,92,246,0.12)"))
    fig_t.add_hline(y=0.68, line_dash="dash", line_color="#DC2626",
                    annotation_text="Crisis")
    fig_t.update_layout(title="GFN Risk Trajectory", height=320,
                        template="plotly_dark",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        yaxis=dict(range=[0.3, 1.0]), showlegend=False)
    st.plotly_chart(fig_t, use_container_width=True)


def _render_hysteresis_panel(df, enter=0.68, exit_=0.60):
    import streamlit as st
    st.subheader("📉 Hysteresis (Crisis Lock-in)")
    if df is None or df.empty or "in_crisis_mode" not in df.columns:
        st.info("Нет данных гистерезиса")
        return
    months, gfn = df["month"], df["gfn_risk"]
    fig = go.Figure()
    fig.add_hrect(y0=enter, y1=1.0,
                  fillcolor="rgba(220,38,38,0.12)", line_width=0)
    fig.add_hrect(y0=0.0, y1=exit_,
                  fillcolor="rgba(16,185,129,0.10)", line_width=0)
    fig.add_trace(go.Scatter(x=months, y=gfn, mode="lines+markers",
                             name="GFN Risk",
                             line=dict(color="#8B5CF6", width=2.5)))
    crisis = df[df["in_crisis_mode"] == 1]
    if not crisis.empty:
        fig.add_trace(go.Scatter(x=crisis["month"], y=crisis["gfn_risk"],
                                 mode="markers", name="In crisis mode",
                                 marker=dict(size=10, color="#DC2626",
                                             symbol="diamond")))
    fig.add_hline(y=enter, line_dash="dash", line_color="#DC2626",
                  annotation_text=f"Enter {enter}")
    fig.add_hline(y=exit_, line_dash="dot", line_color="#10B981",
                  annotation_text=f"Exit {exit_}")
    fig.update_layout(title="Hysteresis: enter ≥ 0.68 / exit ≤ 0.60",
                      height=380, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="white"),
                      yaxis=dict(range=[0.3, 1.0]),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Months in crisis mode", int(df["in_crisis_mode"].sum()))
    c2.metric("Enter", f"{enter:.2f}")
    c3.metric("Exit", f"{exit_:.2f}")


def _render_liquidity_panel(df):
    import streamlit as st
    st.subheader("💧 Liquidity & Credit")
    if df is None or df.empty or "liquidity" not in df.columns:
        st.info("Нет данных ликвидности")
        return
    months, liq, credit = df["month"], df["liquidity"], df["credit_factor"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=months, y=liq, mode="lines+markers",
                             name="Liquidity",
                             line=dict(color="#06B6D4", width=2.5),
                             fill="tozeroy",
                             fillcolor="rgba(6,182,212,0.12)"))
    fig.add_trace(go.Scatter(x=months, y=credit, mode="lines",
                             name="Credit factor",
                             line=dict(color="#F59E0B", width=2, dash="dash")))
    if "liquidity_shock" in df.columns:
        shocks = df[df["liquidity_shock"] == 1]
        if not shocks.empty:
            fig.add_trace(go.Scatter(x=shocks["month"], y=shocks["liquidity"],
                                     mode="markers", name="Shock",
                                     marker=dict(size=12, color="#DC2626",
                                                 symbol="x")))
    fig.add_hline(y=0.40, line_dash="dash", line_color="#F59E0B")
    fig.add_hline(y=0.25, line_dash="dash", line_color="#DC2626")
    fig.update_layout(height=360, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="white"),
                      yaxis=dict(range=[0, 1]),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Liquidity", f"{liq.iloc[-1]:.3f}")
    c2.metric("Credit factor", f"{credit.iloc[-1]:.3f}")
    c3.metric("Shock events",
              int(df["liquidity_shock"].sum()) if "liquidity_shock" in df.columns else 0)


def _render_financial_network(gfn_risk: float = 0.48):
    import streamlit as st
    st.subheader("🕸️ Financial Network (Core–Periphery)")
    nodes = {
        "Fed/US": {"camp": "core", "size": 45}, "ECB/EU": {"camp": "core", "size": 38},
        "BoJ/JP": {"camp": "core", "size": 32}, "PBoC/CN": {"camp": "core", "size": 36},
        "UK": {"camp": "semi", "size": 26}, "Switzerland": {"camp": "semi", "size": 22},
        "Singapore": {"camp": "semi", "size": 20}, "Saudi": {"camp": "semi", "size": 24},
        "Ukraine": {"camp": "peri", "size": 16}, "Turkey": {"camp": "peri", "size": 18},
        "Argentina": {"camp": "peri", "size": 15}, "Nigeria": {"camp": "peri", "size": 14},
        "Pakistan": {"camp": "peri", "size": 14}, "Egypt": {"camp": "peri", "size": 15},
    }
    G = nx.Graph()
    for name, attr in nodes.items():
        G.add_node(name, **attr)
    edges = [
        ("Fed/US", "ECB/EU"), ("Fed/US", "BoJ/JP"), ("Fed/US", "PBoC/CN"),
        ("ECB/EU", "BoJ/JP"), ("Fed/US", "UK"), ("Fed/US", "Switzerland"),
        ("ECB/EU", "UK"), ("Fed/US", "Saudi"), ("Fed/US", "Ukraine"),
        ("ECB/EU", "Ukraine"), ("Fed/US", "Turkey"), ("PBoC/CN", "Pakistan"),
        ("Fed/US", "Argentina"), ("Fed/US", "Egypt"),
    ]
    G.add_edges_from(edges)
    pos = nx.spring_layout(G, seed=42, k=1.8, iterations=80)
    for n in G.nodes():
        if G.nodes[n]["camp"] == "core":
            pos[n] = pos[n] * 0.45
        elif G.nodes[n]["camp"] == "semi":
            pos[n] = pos[n] * 0.75
    camp_color = {"core": "#3B82F6", "semi": "#F59E0B",
                  "peri": "#DC2626" if gfn_risk >= 0.55 else "#F87171"}
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                             line=dict(width=1.2,
                                       color="rgba(148,163,184,0.45)"),
                             hoverinfo="none", showlegend=False))
    for camp, color in camp_color.items():
        ns = [n for n in G.nodes() if G.nodes[n]["camp"] == camp]
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
            mode="markers+text",
            marker=dict(size=[G.nodes[n]["size"] for n in ns], color=color,
                        line=dict(width=2, color="white"), opacity=0.9),
            text=ns, textposition="top center",
            textfont=dict(size=11, color="white"),
            name={"core": "Core", "semi": "Semi", "peri": "Periphery"}[camp]))
    status = ("🔴 Pressure on periphery" if gfn_risk >= 0.60
              else "🟡 Elevated" if gfn_risk >= 0.50 else "🟢 Stable")
    fig.update_layout(title=f"GFN Network • risk={gfn_risk:.3f} | {status}",
                      height=520, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="white"),
                      xaxis=dict(showgrid=False, zeroline=False,
                                 showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False,
                                 showticklabels=False),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)


def _render_escalation_barometer(df, scenario="base"):
    import streamlit as st
    st.subheader("📊 Escalation + Coupling")
    if df is None or df.empty:
        return
    esc = df["escalation"] * 100
    current = float(esc.iloc[-1])
    months = df["month"]
    fig = go.Figure()
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(16,185,129,0.1)", line_width=0)
    fig.add_hrect(y0=30, y1=55, fillcolor="rgba(251,191,36,0.1)", line_width=0)
    fig.add_hrect(y0=55, y1=75, fillcolor="rgba(245,158,11,0.15)", line_width=0)
    fig.add_hrect(y0=75, y1=100, fillcolor="rgba(220,38,38,0.2)", line_width=0)
    fig.add_trace(go.Scatter(x=months, y=esc, mode="lines+markers",
                             name="Escalation",
                             line=dict(color="#DC2626", width=2.5),
                             fill="tozeroy",
                             fillcolor="rgba(220,38,38,0.1)"))
    if "gfn_risk" in df.columns:
        fig.add_trace(go.Scatter(x=months, y=df["gfn_risk"] * 100,
                                 mode="lines", name="GFN×100",
                                 line=dict(color="#8B5CF6", width=2,
                                           dash="dash")))
    fig.update_layout(title=f"Escalation vs GFN • {scenario}",
                      height=360, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="white"),
                      yaxis=dict(range=[0, 105]),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Escalation", f"{current:.1f}%",
              delta=f"{current - float(esc.iloc[0]):+.1f}")
    c2.metric("Status", "Peace" if current < 30 else
              "Tension" if current < 55 else
              "Conflict" if current < 75 else "Crisis")
    if "gfn_extra_shock" in df.columns:
        c3.metric("Avg GFN→Shock", f"{df['gfn_extra_shock'].mean():.4f}")


def _render_nrdcpi_dynamics(df):
    import streamlit as st
    st.subheader("📈 NRDCPI Dynamics")
    if df is None or df.empty:
        return
    months = df["month"]
    filt, raw = df["nrdcpi_filtered"], df["nrdcpi"]
    start = datetime(2026, 1, 1)
    dates = [start + timedelta(days=30 * i) for i in range(len(months))]
    fig = go.Figure()
    std = float(np.std(raw - filt)) + 1.2
    fig.add_trace(go.Scatter(x=dates, y=filt + 1.96 * std, mode="lines",
                             line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=filt - 1.96 * std, mode="lines",
                             line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(59,130,246,0.12)",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=dates, y=filt, mode="lines+markers",
                             name="Filtered",
                             line=dict(color="#3B82F6", width=2.8)))
    fig.add_trace(go.Scatter(x=dates, y=raw, mode="lines", name="Raw",
                             line=dict(color="rgba(59,130,246,0.3)",
                                       width=1, dash="dot")))
    fig.add_hline(y=60, line_dash="dash", line_color="#F59E0B")
    fig.add_hline(y=75, line_dash="dash", line_color="#DC2626")
    fig.update_layout(height=400, template="plotly_dark",
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="white"),
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, use_container_width=True)
    cur = float(filt.iloc[-1])
    risk = ("🔴 CRITICAL" if cur >= 75 else "🟠 HIGH" if cur >= 60
            else "🟡 ELEVATED" if cur >= 45 else "🟢 MODERATE")
    st.info(f"**Risk Level**: {risk} (NRDCPI = {cur:.1f})")


def _render_network_viz(model):
    import streamlit as st
    st.subheader("🌐 Network ABM")
    try:
        net, colors, labels = model.get_network_for_viz()
        if net.number_of_nodes() == 0:
            st.info("Network disabled")
            return
        pos = nx.spring_layout(net, seed=42, k=0.3, iterations=35)
        edge_x, edge_y = [], []
        for u, v in net.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                 line=dict(color="rgba(100,100,130,0.2)",
                                           width=1),
                                 hoverinfo="none", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[pos[i][0] for i in range(len(net))],
            y=[pos[i][1] for i in range(len(net))],
            mode="markers",
            marker=dict(size=6, color=colors,
                        colorscale=[[0, "#3B82F6"], [0.5, "#FFF"],
                                    [1, "#DC2626"]],
                        showscale=True, colorbar=dict(title="Ideology"),
                        opacity=0.85),
            text=[f"{lab} | {c:.2f}" for lab, c in zip(labels, colors)],
            hoverinfo="text", showlegend=False))
        fig.update_layout(height=480, template="plotly_dark",
                          xaxis=dict(showgrid=False, zeroline=False,
                                     showticklabels=False),
                          yaxis=dict(showgrid=False, zeroline=False,
                                     showticklabels=False),
                          paper_bgcolor="rgba(0,0,0,0)",
                          plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="white"),
                          margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.warning(str(e))


# =============================================================================
# ENTRYPOINT
# =============================================================================

def _is_streamlit() -> bool:
    """Определяет, запущен ли скрипт через `streamlit run`."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == "__main__":
    if _is_streamlit():
        _streamlit_main()
    else:
        _cli_main()