import pytest
from agentic_reliability_framework.core.governance.risk_engine import RiskEngine
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    GrantAccessIntent,
    DeployConfigurationIntent,
    ResourceType,
    PermissionLevel,
)


def test_risk_low_cost():
    engine = RiskEngine()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment="dev"
    )
    score, _, _ = engine.calculate_risk(intent, cost_estimate=70.0, policy_violations=[])
    # Compute prior mean for COMPUTE category (1.0,12.0) = 1/13 ≈ 0.0769
    assert score == pytest.approx(0.0769, abs=1e-3)


def test_risk_high_cost():
    engine = RiskEngine()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D16s_v3",
        requester="alice",
        environment="prod"
    )
    score, _, _ = engine.calculate_risk(intent, cost_estimate=560.0, policy_violations=[])
    # Prior mean 0.0769 * context multiplier 1.5 = 0.11535
    assert score == pytest.approx(0.1154, abs=1e-3)


def test_risk_access_grant_admin():
    engine = RiskEngine()
    intent = GrantAccessIntent(
        principal="bob",
        permission_level=PermissionLevel.ADMIN,
        resource_scope="/subscriptions/123",
        requester="alice"
    )
    score, _, _ = engine.calculate_risk(intent, cost_estimate=None, policy_violations=[])
    # Security prior (2.0,10.0) mean = 2/12 ≈ 0.1667
    assert score == pytest.approx(0.1667, abs=1e-3)


def test_risk_with_policy_violations():
    engine = RiskEngine()
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D8s_v3",
        requester="alice",
        environment="prod"
    )
    score, _, _ = engine.calculate_risk(
        intent,
        cost_estimate=280.0,
        policy_violations=["region not allowed", "cost too high"]
    )
    # Prior mean 0.0769 * 1.5 = 0.11535
    assert score == pytest.approx(0.1154, abs=1e-3)


def test_deploy_config_risk():
    engine = RiskEngine()
    intent = DeployConfigurationIntent(
        service_name="api",
        change_scope="global",
        deployment_target="prod",
        requester="alice",
        configuration={}
    )
    score, _, _ = engine.calculate_risk(intent, cost_estimate=None, policy_violations=[])
    # DEFAULT prior (1.0,10.0) mean = 1/11 ≈ 0.0909, times 1.5 = 0.13636
    assert score == pytest.approx(0.1364, abs=1e-3)
