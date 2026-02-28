import pytest
from agentic_reliability_framework.infrastructure.risk_engine import RiskEngine
from agentic_reliability_framework.infrastructure.intents import (
    ProvisionResourceIntent,
    GrantAccessIntent,
    DeployConfigurationIntent,
    ResourceType,
    Environment,
    PermissionLevel,
)


def test_risk_low_cost():
    engine = RiskEngine()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment=Environment.DEV
    )
    score, expl = engine.calculate_risk(intent, cost_estimate=70.0, policy_violations=[])
    assert score == pytest.approx(0.1 + (70/5000)*0.3, rel=1e-2)  # ~0.1042


def test_risk_high_cost():
    engine = RiskEngine()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D16s_v3",
        requester="alice",
        environment=Environment.PROD
    )
    score, expl = engine.calculate_risk(intent, cost_estimate=560.0, policy_violations=[])
    # base 0.1 + cost (560/5000*0.3=0.0336) + env 0.1 = 0.2336
    assert score == pytest.approx(0.23, abs=0.01)


def test_risk_access_grant_admin():
    engine = RiskEngine()
    intent = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.ADMIN,
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    score, expl = engine.calculate_risk(intent, cost_estimate=None, policy_violations=[])
    # intent type 0.3 + permission 0.8*0.3 = 0.24 -> total 0.54
    assert score == 0.54


def test_risk_with_policy_violations():
    engine = RiskEngine()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D8s_v3",
        requester="alice",
        environment=Environment.PROD
    )
    score, expl = engine.calculate_risk(
        intent,
        cost_estimate=280.0,
        policy_violations=["region not allowed", "cost too high"]
    )
    # base 0.1 + cost (280/5000*0.3=0.0168) + env 0.1 + violations 2*0.2=0.4 -> total 0.6168
    assert score == pytest.approx(0.62, abs=0.01)
