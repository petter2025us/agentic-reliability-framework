"""
Root Cause Agent – specializes in root cause analysis.
"""

import logging
from typing import Dict, Any, List
from agentic_reliability_framework.core.models.event import ReliabilityEvent
from agentic_reliability_framework.core.config.constants import (
    LATENCY_EXTREME, LATENCY_CRITICAL,
    ERROR_RATE_WARNING, ERROR_RATE_HIGH, ERROR_RATE_CRITICAL,
    CPU_CRITICAL, MEMORY_CRITICAL, CPU_WARNING, MEMORY_WARNING
)
from .base import BaseAgent, AgentSpecialization

logger = logging.getLogger(__name__)


class RootCauseAgent(BaseAgent):
    def __init__(self):
        super().__init__(AgentSpecialization.DIAGNOSTICIAN)
        logger.info("Initialized RootCauseAgent")

    async def analyze(self, event: ReliabilityEvent) -> Dict[str, Any]:
        try:
            causes = self._analyze_potential_causes(event)
            return {
                'specialization': self.specialization.value,
                'confidence': 0.7,
                'findings': {
                    'likely_root_causes': causes,
                    'evidence_patterns': self._identify_evidence(event),
                    'investigation_priority': self._prioritize_investigation(causes)
                },
                'recommendations': [f"Check {cause['cause']} for issues" for cause in causes[:2]]
            }
        except Exception as e:
            logger.error(f"RootCauseAgent error: {e}", exc_info=True)
            return {'specialization': self.specialization.value, 'confidence': 0.0, 'findings': {}, 'recommendations': [f"Analysis error: {str(e)}"]}

    def _analyze_potential_causes(self, event: ReliabilityEvent) -> List[Dict[str, Any]]:
        causes = []
        if event.latency_p99 > LATENCY_EXTREME and event.error_rate > 0.2:
            causes.append({
                "cause": "Database/External Dependency Failure",
                "confidence": 0.85,
                "evidence": f"Extreme latency ({event.latency_p99:.0f}ms) with high errors ({event.error_rate*100:.1f}%)",
                "investigation": "Check database connection pool, external API health"
            })
        if (event.cpu_util and event.cpu_util > CPU_CRITICAL and
            event.memory_util and event.memory_util > MEMORY_CRITICAL):
            causes.append({
                "cause": "Resource Exhaustion",
                "confidence": 0.90,
                "evidence": f"CPU ({event.cpu_util*100:.1f}%) and Memory ({event.memory_util*100:.1f}%) critically high",
                "investigation": "Check for memory leaks, infinite loops, insufficient resources"
            })
        if event.error_rate > ERROR_RATE_CRITICAL and event.latency_p99 < 200:
            causes.append({
                "cause": "Application Bug / Configuration Issue",
                "confidence": 0.75,
                "evidence": f"High error rate ({event.error_rate*100:.1f}%) without latency impact",
                "investigation": "Review recent deployments, configuration changes, application logs"
            })
        if (200 <= event.latency_p99 <= 400 and
            ERROR_RATE_WARNING <= event.error_rate <= ERROR_RATE_HIGH):
            causes.append({
                "cause": "Gradual Performance Degradation",
                "confidence": 0.65,
                "evidence": f"Moderate latency ({event.latency_p99:.0f}ms) and errors ({event.error_rate*100:.1f}%)",
                "investigation": "Check resource trends, dependency performance, capacity planning"
            })
        if not causes:
            causes.append({
                "cause": "Unknown - Requires Investigation",
                "confidence": 0.3,
                "evidence": "Pattern does not match known failure modes",
                "investigation": "Complete system review needed"
            })
        return causes

    def _identify_evidence(self, event: ReliabilityEvent) -> List[str]:
        evidence = []
        if event.latency_p99 > event.error_rate * 1000:
            evidence.append("latency_disproportionate_to_errors")
        if (event.cpu_util and event.cpu_util > CPU_WARNING and
            event.memory_util and event.memory_util > MEMORY_WARNING):
            evidence.append("correlated_resource_exhaustion")
        if event.error_rate > ERROR_RATE_HIGH and event.latency_p99 < LATENCY_CRITICAL:
            evidence.append("errors_without_latency_impact")
        return evidence

    def _prioritize_investigation(self, causes: List[Dict[str, Any]]) -> str:
        for cause in causes:
            if "Database" in cause["cause"] or "Resource Exhaustion" in cause["cause"]:
                return "HIGH"
        return "MEDIUM"
