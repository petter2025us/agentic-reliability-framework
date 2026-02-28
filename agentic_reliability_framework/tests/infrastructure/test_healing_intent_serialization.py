import pytest
import time
from agentic_reliability_framework.core.governance.healing_intent import (
    HealingIntent as InfraHealingIntent,
    IntentSource,
    IntentStatus,
    RecommendedAction,
    ConfidenceDistribution,
)
from agentic_reliability_framework import (
    HealingIntentSerializer as OldSerializer,
    HealingIntent as OldHealingIntent,
    IntentSource as OldIntentSource,
    IntentStatus as OldIntentStatus,
)


@pytest.fixture
def sample_infra_healing_intent():
    """Create a fully populated infrastructure HealingIntent."""
    conf_dist = ConfidenceDistribution(0.85, 0.05)
    return InfraHealingIntent(
        action="provision_vm",
        component="web-server",
        parameters={"size": "Standard_D2s_v3", "region": "eastus"},
        justification="Scale out due to high CPU",
        confidence=0.85,
        confidence_distribution=conf_dist.to_dict(),
        incident_id="inc_12345",
        detected_at=time.time(),
        risk_score=0.35,
        risk_factors={"cost": 0.2, "permission": 0.15},
        cost_projection=120.0,
        cost_confidence_interval=(110.0, 130.0),
        recommended_action=RecommendedAction.APPROVE,
        decision_tree=[{"step": "check_policy", "outcome": "pass"}],
        alternative_actions=[{"action": "scale_out", "confidence": 0.7}],
        reasoning_chain=["CPU > 80%", "Memory > 75%"],
        similar_incidents=[{"id": "inc_123", "similarity": 0.9}],
        rag_similarity_score=0.85,
        source=IntentSource.INFRASTRUCTURE_ANALYSIS,
        status=IntentStatus.OSS_ADVISORY_ONLY,
        policy_violations=["region_not_allowed"],
        infrastructure_intent_id="infra_001",
    )


def test_serialization_roundtrip(sample_infra_healing_intent):
    """
    Verify that an infrastructure HealingIntent can be serialized to the
    enterprise request format and then deserialized by the old ARF's
    HealingIntentSerializer without loss of core fields.
    """
    # 1. Convert infra intent to enterprise request dict
    enterprise_req = sample_infra_healing_intent.to_enterprise_request()

    # 2. Use old serializer to deserialize into an old HealingIntent
    #    (the old serializer expects the versioned wrapper, so we wrap it)
    versioned_data = {
        "version": "2.0.0",
        "data": enterprise_req,
        "metadata": {"serialized_at": time.time()}
    }
    old_intent = OldSerializer.deserialize(versioned_data)

    # 3. Assert core fields match
    assert old_intent.action == sample_infra_healing_intent.action
    assert old_intent.component == sample_infra_healing_intent.component
    assert old_intent.parameters == sample_infra_healing_intent.parameters
    assert old_intent.justification == sample_infra_healing_intent.justification
    assert old_intent.confidence == sample_infra_healing_intent.confidence
    assert old_intent.incident_id == sample_infra_healing_intent.incident_id
    assert abs(old_intent.detected_at - sample_infra_healing_intent.detected_at) < 1.0

    # 4. Verify OSS metadata was preserved
    oss_meta = enterprise_req["oss_metadata"]
    assert oss_meta["risk_score"] == sample_infra_healing_intent.risk_score
    assert oss_meta["policy_violations_count"] == len(sample_infra_healing_intent.policy_violations)
    assert oss_meta["confidence_basis"] == sample_infra_healing_intent._get_confidence_basis()

    # 5. Ensure the old intent is marked as requiring enterprise
    assert old_intent.requires_enterprise is True


def test_oss_advisory_flag_preserved(sample_infra_healing_intent):
    """
    Ensure that the OSS_ADVISORY_ONLY status is carried through in the
    enterprise request (as part of oss_metadata) and can be recreated.
    """
    enterprise_req = sample_infra_healing_intent.to_enterprise_request()
    assert enterprise_req["oss_metadata"]["is_oss_advisory"] is True
    assert enterprise_req["execution_allowed"] is False
    assert enterprise_req["requires_enterprise"] is True
