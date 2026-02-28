"""
Orchestration Manager – coordinates multiple specialized agents.
"""

import asyncio
import logging
from typing import Dict, Any, Optional
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
    def __init__(self, detective: Optional[AnomalyDetectionAgent] = None,
                 diagnostician: Optional[RootCauseAgent] = None,
                 predictive: Optional[PredictiveAgent] = None):
        self.agents = {
            AgentSpecialization.DETECTIVE: detective or AnomalyDetectionAgent(),
            AgentSpecialization.DIAGNOSTICIAN: diagnostician or RootCauseAgent(),
            AgentSpecialization.PREDICTIVE: predictive or PredictiveAgent(SimplePredictiveEngine()),
        }
        logger.info(f"Initialized OrchestrationManager with {len(self.agents)} agents")

    async def orchestrate_analysis(self, event: ReliabilityEvent) -> Dict[str, Any]:
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
        return self._synthesize_agent_findings(event, agent_results)

    def _synthesize_agent_findings(self, event: ReliabilityEvent, agent_results: Dict) -> Dict[str, Any]:
        detective_result = agent_results.get(AgentSpecialization.DETECTIVE.value)
        diagnostician_result = agent_results.get(AgentSpecialization.DIAGNOSTICIAN.value)
        predictive_result = agent_results.get(AgentSpecialization.PREDICTIVE.value)
        if not detective_result:
            logger.warning("No detective agent results available")
            return {'error': 'No agent results available'}
        synthesis = {
            'incident_summary': {
                'severity': detective_result['findings'].get('severity_tier', 'UNKNOWN'),
                'anomaly_confidence': detective_result['confidence'],
                'primary_metrics_affected': [m["metric"] for m in detective_result['findings'].get('primary_metrics_affected', [])]
            },
            'root_cause_insights': diagnostician_result['findings'] if diagnostician_result else {},
            'predictive_insights': predictive_result['findings'] if predictive_result else {},
            'recommended_actions': self._prioritize_actions(
                detective_result.get('recommendations', []),
                diagnostician_result.get('recommendations', []) if diagnostician_result else [],
                predictive_result.get('recommendations', []) if predictive_result else []
            ),
            'agent_metadata': {
                'participating_agents': list(agent_results.keys()),
                'analysis_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat()
            }
        }
        return synthesis

    def _prioritize_actions(self, detection: List[str], diagnosis: List[str], predictive: List[str]) -> List[str]:
        all_actions = detection + diagnosis + predictive
        seen = set()
        unique = []
        for a in all_actions:
            if a not in seen:
                seen.add(a)
                unique.append(a)
        return unique[:5]
