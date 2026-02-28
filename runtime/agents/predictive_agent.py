"""
Predictive Agent – specializes in forecasting and trend analysis.
"""

import logging
from typing import Dict, Any
from agentic_reliability_framework.core.models.event import ReliabilityEvent
from agentic_reliability_framework.runtime.analytics.predictive import SimplePredictiveEngine
from .base import BaseAgent, AgentSpecialization

logger = logging.getLogger(__name__)


class PredictiveAgent(BaseAgent):
    def __init__(self, engine: SimplePredictiveEngine):
        super().__init__(AgentSpecialization.PREDICTIVE)
        self.engine = engine
        logger.info("Initialized PredictiveAgent")

    async def analyze(self, event: ReliabilityEvent) -> Dict[str, Any]:
        try:
            event_data = {
                'latency_p99': event.latency_p99,
                'error_rate': event.error_rate,
                'throughput': event.throughput,
                'cpu_util': event.cpu_util,
                'memory_util': event.memory_util
            }
            self.engine.add_telemetry(event.component, event_data)
            insights = self.engine.get_predictive_insights(event.component)
            return {
                'specialization': self.specialization.value,
                'confidence': 0.8 if insights['critical_risk_count'] > 0 else 0.5,
                'findings': insights,
                'recommendations': insights['recommendations']
            }
        except Exception as e:
            logger.error(f"PredictiveAgent error: {e}", exc_info=True)
            return {'specialization': self.specialization.value, 'confidence': 0.0, 'findings': {}, 'recommendations': [f"Analysis error: {str(e)}"]}
