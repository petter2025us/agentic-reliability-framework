import pytest
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine
from agentic_reliability_framework.core.governance.intents import (
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
    score, _, _ = engine.calculate_risk(intent, cost_estimate=70.0, policy_violations=[])
    # expected: intent_type 0.1 + cost (70/5000*0.3=0.0042) = 0.1042
    assert score == pytest.approx(0.1042, abs=1e-3)


def test_risk_high_cost():
    engine = RiskEngine()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D16s_v3",
        requester="alice",
        environment=Environment.PROD
    )
    score, _, _ = engine.calculate_risk(intent, cost_estimate=560.0, policy_violations=[])
    # base 0.1 + cost (560/5000*0.3=0.0336) + env 0.1 = 0.2336
    assert score == pytest.approx(0.2336, abs=1e-3)


def test_risk_access_grant_admin():
    engine = RiskEngine()
    intent = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.ADMIN,
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    score, _, _ = engine.calculate_risk(intent, cost_estimate=None, policy_violations=[])
    # intent_type 0.3 + permission (0.8*0.3=0.24) = 0.54
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
    score, _, _ = engine.calculate_risk(
        intent,
        cost_estimate=280.0,
        policy_violations=["region not allowed", "cost too high"]
    )
    # base 0.1 + cost (280/5000*0.3=0.0168) + env 0.1 + violations (2*0.2=0.4) = 0.6168
    assert score == pytest.approx(0.6168, abs=1e-3)


def test_deploy_config_risk():
    engine = RiskEngine()
    intent = DeployConfigurationIntent(
        service_name="api",
        change_scope="global",
        deployment_target=Environment.PROD,
        requester="alice",
        configuration={}
    )
    score, _, _ = engine.calculate_risk(intent, cost_estimate=None, policy_violations=[])
    # intent_type 0.2 + scope 0.6*0.2 = 0.12 + env 0.1 = 0.42
    assert score == 0.42
