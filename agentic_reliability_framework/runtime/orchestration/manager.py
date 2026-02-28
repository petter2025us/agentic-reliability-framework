"""
Orchestration Manager – coordinates multiple specialized agents with uncertainty-aware synthesis.
"""

import asyncio
import logging
import datetime
from typing import Dict, Any, Optional, List, Tuple
from collections import defaultdict
from circuitbreaker import circuit

from agentic_reliability_framework.core.models.event import ReliabilityEvent
from agentic_reliability_framework.core.config.constants import AGENT_TIMEOUT_SECONDS
from agentic_reliability_framework.runtime.agents.base import BaseAgent, AgentSpecialization
from agentic_reliability_framework.runtime.agents.detective import AnomalyDetectionAgent
from agentic_reliability_framework.runtime.agents.diagnostician import RootCauseAgent
from agentic_reliability_framework.runtime.agents.predictive_agent import PredictiveAgent
from agentic_reliability_framework.runtime.analytics.predictive import SimplePredictiveEngine

logger = logging.getLogger(__name__)


@circuit(failure_threshold=3, recovery_timeout=30, name="agent_circuit_breaker")
async def call_agent_with_protection(agent: BaseAgent, event: ReliabilityEvent) -> Dict[str, Any]:
    try:
        result = await asyncio.wait_for(agent.analyze(event), timeout=AGENT_TIMEOUT_SECONDS)
        return result
    except asyncio.TimeoutError:
        logger.warning(f"Agent {agent.specialization.value} timed out")
        raise
    except Exception as e:
        logger.error(f"Agent {agent.specialization.value} error: {e}", exc_info=True)
        raise


class OrchestrationManager:
    """
    Orchestrates multiple agents and synthesizes their outputs with uncertainty weighting.
    """

    def __init__(self, detective: Optional[AnomalyDetectionAgent] = None,
                 diagnostician: Optional[RootCauseAgent] = None,
                 predictive: Optional[PredictiveAgent] = None):
        self.agents = {
            AgentSpecialization.DETECTIVE: detective or AnomalyDetectionAgent(),
            AgentSpecialization.DIAGNOSTICIAN: diagnostician or RootCauseAgent(),
            AgentSpecialization.PREDICTIVE: predictive or PredictiveAgent(SimplePredictiveEngine()),
        }
        # Agent reliability weights (could be learned over time)
        self.agent_reliability = {
            AgentSpecialization.DETECTIVE: 1.0,
            AgentSpecialization.DIAGNOSTICIAN: 1.0,
            AgentSpecialization.PREDICTIVE: 1.0,
        }
        logger.info(f"Initialized OrchestrationManager with {len(self.agents)} agents")

    async def orchestrate_analysis(self, event: ReliabilityEvent) -> Dict[str, Any]:
        """
        Run all agents in parallel, collect results, and synthesize with uncertainty.
        Returns a dictionary containing:
            - incident_summary
            - root_cause_insights
            - predictive_insights
            - recommended_actions (weighted by agent confidence and reliability)
            - agent_metadata
            - uncertainty_quantification (combined confidence intervals)
        """
        agent_tasks = []
        agent_specs = []
        for spec, agent in self.agents.items():
            agent_tasks.append(call_agent_with_protection(agent, event))
            agent_specs.append(spec)
        agent_results = {}
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*agent_tasks, return_exceptions=True),
                timeout=AGENT_TIMEOUT_SECONDS + 1
            )
            for spec, result in zip(agent_specs, results):
                if isinstance(result, Exception):
                    logger.error(f"Agent {spec.value} failed: {result}")
                    continue
                agent_results[spec.value] = result
        except asyncio.TimeoutError:
            logger.warning("Agent orchestration timed out")
        except Exception as e:
            logger.error(f"Agent orchestration error: {e}", exc_info=True)

        # If no agents succeeded, return error
        if not agent_results:
            return {'error': 'No agent results available'}

        # Synthesize with uncertainty weighting
        synthesis = self._synthesize_agent_findings_weighted(event, agent_results)
        return synthesis

    def _synthesize_agent_findings_weighted(self, event: ReliabilityEvent,
                                             agent_results: Dict) -> Dict[str, Any]:
        """
        Combine agent findings using confidence-weighted voting.
        """
        detective_result = agent_results.get(AgentSpecialization.DETECTIVE.value)
        diagnostician_result = agent_results.get(AgentSpecialization.DIAGNOSTICIAN.value)
        predictive_result = agent_results.get(AgentSpecialization.PREDICTIVE.value)

        # Defaults if missing
        if not detective_result:
            logger.warning("Detective agent missing, using fallback")
            detective_result = {
                'findings': {'severity_tier': 'UNKNOWN', 'primary_metrics_affected': []},
                'confidence': 0.0,
                'recommendations': []
            }

        # Weighted action voting
        action_weights = defaultdict(float)
        action_details = []

        # Process detective recommendations
        if detective_result:
            conf = detective_result.get('confidence', 0.0)
            reliability = self.agent_reliability.get(AgentSpecialization.DETECTIVE, 1.0)
            weight = conf * reliability
            for rec in detective_result.get('recommendations', []):
                action_weights[rec] += weight
                action_details.append({
                    'action': rec,
                    'weight': weight,
                    'source': 'detective',
                    'confidence': conf
                })

        # Process diagnostician recommendations
        if diagnostician_result:
            conf = diagnostician_result.get('confidence', 0.7)  # default if missing
            reliability = self.agent_reliability.get(AgentSpecialization.DIAGNOSTICIAN, 1.0)
            weight = conf * reliability
            for rec in diagnostician_result.get('recommendations', []):
                action_weights[rec] += weight
                action_details.append({
                    'action': rec,
                    'weight': weight,
                    'source': 'diagnostician',
                    'confidence': conf
                })

        # Process predictive recommendations
        if predictive_result:
            conf = predictive_result.get('confidence', 0.5)
            reliability = self.agent_reliability.get(AgentSpecialization.PREDICTIVE, 1.0)
            weight = conf * reliability
            for rec in predictive_result.get('recommendations', []):
                action_weights[rec] += weight
                action_details.append({
                    'action': rec,
                    'weight': weight,
                    'source': 'predictive',
                    'confidence': conf
                })

        # Sort actions by total weight descending
        sorted_actions = sorted(action_weights.items(), key=lambda x: x[1], reverse=True)
        top_actions = [action for action, _ in sorted_actions[:5]]

        # Compute overall confidence interval for the top action (if any)
        uncertainty = {}
        if sorted_actions:
            top_action = sorted_actions[0][0]
            weights_for_top = [d['weight'] for d in action_details if d['action'] == top_action]
            if weights_for_top:
                # Simple uncertainty: range of weights (could be refined)
                uncertainty['top_action_weight_range'] = (min(weights_for_top), max(weights_for_top))
                uncertainty['top_action_mean_weight'] = sum(weights_for_top) / len(weights_for_top)

        # Build synthesis
        synthesis = {
            'incident_summary': {
                'severity': detective_result['findings'].get('severity_tier', 'UNKNOWN'),
                'anomaly_confidence': detective_result.get('confidence', 0.0),
                'primary_metrics_affected': detective_result['findings'].get('primary_metrics_affected', [])
            },
            'root_cause_insights': diagnostician_result['findings'] if diagnostician_result else {},
            'predictive_insights': predictive_result['findings'] if predictive_result else {},
            'recommended_actions': top_actions,
            'action_details': action_details,
            'agent_metadata': {
                'participating_agents': list(agent_results.keys()),
                'analysis_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
            },
            'uncertainty_quantification': uncertainty
        }
        return synthesis

    def update_agent_reliability(self, agent_specialization: AgentSpecialization,
                                  success: bool):
        """
        Update agent reliability based on historical success (Bayesian update).
        Simple moving average for now; can be replaced with beta distribution.
        """
        current = self.agent_reliability.get(agent_specialization, 1.0)
        # Exponential moving average
        alpha = 0.1
        if success:
            self.agent_reliability[agent_specialization] = (1 - alpha) * current + alpha * 1.0
        else:
            self.agent_reliability[agent_specialization] = (1 - alpha) * current + alpha * 0.0
        logger.info(f"Updated reliability for {agent_specialization.value}: {self.agent_reliability[agent_specialization]:.3f}")
