"""
Azure Infrastructure Simulator – main entry point for evaluating infrastructure intents.

Orchestrates cost estimation, policy evaluation, and risk scoring to produce a HealingIntent.
"""

from typing import List

from agentic_reliability_framework.infrastructure.intents import InfrastructureIntent
from agentic_reliability_framework.infrastructure.policies import Policy, PolicyEvaluator
from agentic_reliability_framework.infrastructure.cost_estimator import CostEstimator
from agentic_reliability_framework.infrastructure.risk_engine import RiskEngine
from agentic_reliability_framework.infrastructure.healing_intent import HealingIntent, RecommendedAction


class AzureInfrastructureSimulator:
    """
    Simulates Azure infrastructure governance.

    This class is the core of the OSS advisory module. It evaluates an infrastructure
    intent against policies, estimates cost, calculates risk, and returns a HealingIntent
    with a recommended action.
    """

    def __init__(self, policies: List[Policy]):
        """
        Initialize the simulator with a list of policies.

        Args:
            policies: List of Policy objects to enforce.
        """
        self.policy_evaluator = PolicyEvaluator(policies)
        self.cost_estimator = CostEstimator()
        self.risk_engine = RiskEngine()

    def evaluate(self, intent: InfrastructureIntent) -> HealingIntent:
        """
        Evaluate an infrastructure intent and return a HealingIntent.

        The evaluation is deterministic and has no side effects.
        """
        # Step 1: Estimate cost (if applicable)
        cost = None
        if hasattr(intent, "resource_type"):  # ProvisionResourceIntent
            cost = self.cost_estimator.estimate_monthly_cost(intent)  # type: ignore

        # Step 2: Evaluate policies
        violations = self.policy_evaluator.evaluate(intent)

        # Step 3: Calculate risk score
        risk_score, explanation = self.risk_engine.calculate_risk(intent, cost, violations)

        # Step 4: Determine recommended action
        # Rule: if risk > 0.8 or any policy violations -> DENY
        #       if risk > 0.4 -> ESCALATE
        #       else -> APPROVE
        if risk_score > 0.8 or violations:
            recommended_action = RecommendedAction.DENY
        elif risk_score > 0.4:
            recommended_action = RecommendedAction.ESCALATE
        else:
            recommended_action = RecommendedAction.APPROVE

        # Step 5: Build justification
        justification_parts = [f"Risk score: {risk_score:.2f}."]
        if cost is not None:
            justification_parts.append(f"Estimated monthly cost: ${cost:.2f}.")
        if violations:
            justification_parts.append(f"Policy violations: {'; '.join(violations)}.")
        justification_parts.append(explanation)
        justification = " ".join(justification_parts)

        # Step 6: Create summary
        intent_summary = f"{intent.intent_type} requested by {intent.requester}"

        return HealingIntent(
            intent_summary=intent_summary,
            cost_projection=cost,
            risk_score=risk_score,
            policy_violations=violations,
            recommended_action=recommended_action,
            justification=justification,
            confidence_score=0.9,  # could be tuned based on data quality
        )
