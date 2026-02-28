import pytest
from datetime import datetime
from agentic_reliability_framework.core.governance.intents import (
    ProvisionResourceIntent,
    DeployConfigurationIntent,
    GrantAccessIntent,
    ResourceType,
    Environment,
    PermissionLevel,
)


def test_provision_resource_intent_creation():
    intent = ProvisionResourceIntent(
        resource_type=ResourceType.VM,
        region="eastus",
        size="Standard_D2s_v3",
        requester="alice",
        environment=Environment.PROD
    )
    assert intent.intent_type == "provision_resource"
    assert intent.requester == "alice"
    assert intent.environment == Environment.PROD


def test_deploy_configuration_intent_creation():
    intent = DeployConfigurationIntent(
        service_name="api",
        change_scope="canary",
        deployment_target=Environment.PROD,
        requester="bob",
        configuration={"feature_x": True}
    )
    assert intent.intent_type == "deploy_config"
    assert intent.risk_level_hint is None


def test_grant_access_intent_creation():
    intent = GrantAccessIntent(
        principal="user:charlie",
        permission_level=PermissionLevel.WRITE,
        resource_scope="/subscriptions/123/resourceGroups/rg",
        requester="dave",
        justification="Need write access for debugging"
    )
    assert intent.intent_type == "grant_access"
    assert intent.permission_level == PermissionLevel.WRITE
