import pytest
from agentic_reliability_framework.infrastructure.azure.azure_simulator import AzureInfrastructureSimulator
from agentic_reliability_framework.infrastructure.policies import Policy
from agentic_reliability_framework.infrastructure.intents import (
    ProvisionResourceIntent,
    GrantAccessIntent,
    ResourceType,
    Environment,
    PermissionLevel,
)
from agentic_reliability_framework.infrastructure.healing_intent import RecommendedAction


def test_simulator_approve_low_risk():
    policies = [Policy(name="allow all", description="no restrictions")]
    simulator = AzureInfrastructureSimulator(policies)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment=Environment.DEV
    )
    result = simulator.evaluate(intent)
    assert result.recommended_action == RecommendedAction.APPROVE
    assert result.risk_score < 0.4
    assert result.cost_projection == 70.0


def test_simulator_deny_high_risk():
    policies = [Policy(name="no admin", max_permission_level="write")]
    simulator = AzureInfrastructureSimulator(policies)
    intent = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.ADMIN,
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    result = simulator.evaluate(intent)
    assert result.recommended_action == RecommendedAction.DENY
    assert result.risk_score > 0.8
    assert "admin" in result.justification


def test_simulator_deny_due_to_cost_threshold():
    policies = [Policy(name="cost limit", cost_threshold_usd=100.0)]
    simulator = AzureInfrastructureSimulator(policies)
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D8s_v3",  # cost 280 > 100
        requester="alice",
        environment=Environment.DEV
    )
    result = simulator.evaluate(intent)
    assert result.recommended_action == RecommendedAction.DENY
    assert "exceeds threshold" in result.justification
    assert len(result.policy_violations) == 1


def test_simulator_escalate_medium_risk():
    policies = [Policy(name="allow all", description="no restrictions")]
    simulator = AzureInfrastructureSimulator(policies)
    # Grant access with write permission gives risk ~0.42
    intent = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.WRITE,
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    result = simulator.evaluate(intent)
    assert result.recommended_action == RecommendedAction.ESCALATE
    assert 0.4 < result.risk_score < 0.8
