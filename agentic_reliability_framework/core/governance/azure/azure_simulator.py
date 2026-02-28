# agentic_reliability_framework/infrastructure/azure/azure_simulator.py
"""
Azure Infrastructure Simulator – Main orchestration engine.

This module ties together intents, policies, cost estimation, and risk scoring
to produce a HealingIntent. It is the primary entry point for the OSS advisory layer.

The simulator is designed to be deterministic, side-effect-free, and easily
extendable by the enterprise layer (which will replace simulation with actual
Azure API calls while preserving the same interface).
"""

from typing import List, Optional, Dict, Any

from agentic_reliability_framework.core.governance.intents import (
    InfrastructureIntent,
    ProvisionResourceIntent,
)
from agentic_reliability_framework.core.governance.policies import (
    Policy,
    PolicyEvaluator,
    CostThresholdPolicy,
)
from agentic_reliability_framework.core.governance.cost_estimator import CostEstimator
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine
from agentic_reliability_framework.core.governance.healing_intent import (
    HealingIntent,
    RecommendedAction,
    IntentSource,
)
from agentic_reliability_framework.constants import MAX_POLICY_VIOLATIONS


class AzureInfrastructureSimulator:
    """
    Orchestrates the evaluation of an infrastructure intent.

    The simulator uses:
        - A policy evaluator (with a policy tree)
        - A cost estimator
        - A risk engine

    It returns a HealingIntent with a recommendation, already marked as OSS advisory.
    """

    def __init__(
        self,
        policy: Policy,
        pricing_file: Optional[str] = None,
        risk_factors: Optional[List] = None,
    ):
        """
        Initialize the simulator.

        Args:
            policy: The root policy (a Policy object, possibly composite).
            pricing_file: Optional path to custom pricing YAML.
            risk_factors: Optional list of custom risk factors.
        """
        self._policy_evaluator = PolicyEvaluator(policy)
        self._cost_estimator = CostEstimator(pricing_file)
        self._risk_engine = RiskEngine(risk_factors if risk_factors else RiskEngine.DEFAULT_FACTORS)

    def evaluate(self, intent: InfrastructureIntent) -> HealingIntent:
        """
        Evaluate the intent and produce a HealingIntent.

        This method is pure and deterministic (same inputs → same output).
        The returned HealingIntent is already marked as OSS advisory.
        """
        # 1. Estimate cost (if applicable)
        cost = None
        if isinstance(intent, ProvisionResourceIntent):
            cost = self._cost_estimator.estimate_monthly_cost(intent)

        # 2. Evaluate policies with context (cost)
        context = {"cost_estimate": cost} if cost is not None else {}
        violations = self._policy_evaluator.evaluate(intent, context)

        # Enforce OSS limit on policy violations
        if len(violations) > MAX_POLICY_VIOLATIONS:
            violations = violations[:MAX_POLICY_VIOLATIONS]

        # 3. Compute risk
        risk_score, explanation, contributions = self._risk_engine.calculate_risk(
            intent, cost, violations
        )

        # 4. Determine recommended action
        #    This is a decision rule; can be made configurable.
        if risk_score > 0.8 or violations:
            recommended_action = RecommendedAction.DENY
        elif risk_score > 0.4:
            recommended_action = RecommendedAction.ESCALATE
        else:
            recommended_action = RecommendedAction.APPROVE

        # 5. Build justification
        justification_parts = [f"Risk score: {risk_score:.2f}."]
        if cost is not None:
            justification_parts.append(f"Estimated monthly cost: ${cost:.2f}.")
        if violations:
            justification_parts.append(f"Policy violations: {'; '.join(violations)}.")
        justification_parts.append(explanation)
        justification = " ".join(justification_parts)

        # 6. Create summary
        intent_summary = f"{intent.intent_type} requested by {intent.requester}"

        # 7. Package evaluation details
        details = {
            "cost_estimate": cost,
            "violations": violations,
            "risk_score": risk_score,
            "factor_contributions": contributions,
        }

        # 8. Create the HealingIntent with proper source and then mark as OSS advisory
        healing_intent = HealingIntent(
            intent_id=intent.intent_id,
            intent_summary=intent_summary,
            cost_projection=cost,
            risk_score=risk_score,
            policy_violations=violations,
            recommended_action=recommended_action,
            justification=justification,
            confidence_score=0.9,  # could be derived from factor uncertainties
            evaluation_details=details,
            source=IntentSource.INFRASTRUCTURE_ANALYSIS,
        )

        # Mark as OSS advisory (sets status=OSS_ADVISORY_ONLY and execution_allowed=False)
        return healing_intent.mark_as_oss_advisory()
