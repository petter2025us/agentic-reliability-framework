import pytest
import json
from agentic_reliability_framework.core.governance.healing_intent import (
    HealingIntent,
    ValidationError,
    IntentStatus,
)


def make_valid_intent():
    return HealingIntent(
        action="restart_service",
        component="webapi",
        parameters={"wait": 30},
        justification="Autoscale triggered",
        confidence=0.5,
    )


def test_valid_intent_passes():
    intent = make_valid_intent()
    assert intent.status == IntentStatus.CREATED


def test_empty_action_raises():
    with pytest.raises(ValidationError):
        HealingIntent(
            action="",
            component="comp",
            parameters={},
            justification="ok",
            confidence=0.5,
        )


def test_justification_too_long():
    with pytest.raises(ValidationError):
        HealingIntent(
            action="a",
            component="c",
            parameters={},
            justification="x" * (HealingIntent.MAX_JUSTIFICATION_LENGTH + 1),
            confidence=0.5,
        )


def test_too_many_parameters():
    with pytest.raises(ValidationError):
        HealingIntent(
            action="a",
            component="c",
            parameters={str(i): i for i in range(HealingIntent.MAX_PARAMETERS_SIZE + 1)},
            justification="ok",
            confidence=0.5,
        )


def test_mark_as_oss_advisory_sets_status():
    intent = make_valid_intent()
    new = intent.mark_as_oss_advisory()
    assert new.status == IntentStatus.OSS_ADVISORY_ONLY
    assert new.requires_enterprise
