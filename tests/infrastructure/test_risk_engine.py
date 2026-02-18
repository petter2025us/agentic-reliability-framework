import pytest
from agentic_reliability_framework.infrastructure.risk_engine import RiskEngine
from agentic_reliability_framework.infrastructure.intents import (
    ProvisionResourceIntent,
    GrantAccessIntent,
    DeployConfigurationIntent,
    ResourceType,
)


def test_risk_low_cost():
    engine = RiskEngine()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment="prod"
    )
    score, expl = engine.calculate_risk(intent, cost_estimate=70.0, policy_violations=[])
    # Base 0.1 + cost factor (70/5000*0.3 ≈ 0.004) ≈ 0.104 -> 0.1 after rounding
    assert score == 0.1


def test_risk_high_cost():
    engine = RiskEngine()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D16s_v3",
        requester="alice",
        environment="prod"
    )
    score, expl = engine.calculate_risk(intent, cost_estimate=560.0, policy_violations=[])
    # base 0.1 + cost factor (560/5000*0.3 = 0.0336) = 0.1336 -> 0.13
    assert score == 0.13


def test_risk_access_grant_admin():
    engine = RiskEngine()
    intent = GrantAccessIntent(
        principal="bob",
        permission_level="admin",
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    score, expl = engine.calculate_risk(intent, cost_estimate=None, policy_violations=[])
    # intent type 0.3 + permission level 0.8*0.3 = 0.24 -> total 0.54 -> 0.54
    assert score == 0.54


def test_risk_with_policy_violations():
    engine = RiskEngine()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D8s_v3",
        requester="alice",
        environment="prod"
    )
    score, expl = engine.calculate_risk(intent, cost_estimate=280.0, policy_violations=["region not allowed", "cost too high"])
    # base 0.1 + cost (280/5000*0.3=0.0168) + violations (2*0.2=0.4) = 0.5168 -> 0.52
    assert score == 0.52
