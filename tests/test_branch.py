from __future__ import annotations

import pytest
from pydantic import BaseModel

from n00dles import BranchError, agent, branch, pipeline, run
from n00dles.core.pipeline import BranchAgent

from .conftest import FakeProvider


class Classification(BaseModel):
    category: str
    confidence: float = 1.0


def test_branch_function_builds_branch_agent():
    provider = FakeProvider()

    @agent(model="fake-model", provider=provider)
    def billing(ticket: str) -> str:
        """Billing."""

    @agent(model="fake-model", provider=provider)
    def support(ticket: str) -> str:
        """Support."""

    b = branch(billing=billing, support=support, default=support)
    assert isinstance(b, BranchAgent)
    assert set(b.routes.keys()) == {"billing", "support"}
    assert b.default is support


def test_branch_requires_at_least_one_route():
    with pytest.raises(ValueError, match="at least one route"):
        branch(default=None)


def test_routes_on_plain_string_output():
    provider = FakeProvider(["billing", "billing reply text"])

    @agent(model="fake-model", provider=provider)
    def classify(ticket: str) -> str:
        """Classify."""

    @agent(model="fake-model", provider=provider)
    def handle_billing(ticket: str) -> str:
        """Handle billing."""

    @agent(model="fake-model", provider=provider)
    def handle_support(ticket: str) -> str:
        """Handle support."""

    triage = pipeline(classify >> branch(billing=handle_billing, support=handle_support))
    result = run(triage, ticket="my invoice is wrong")

    assert result.output == "billing reply text"
    assert [t.name for t in result.agent_traces] == ["classify", "handle_billing"]


def test_routes_on_dict_output_using_category_key():
    import json

    classification = json.dumps({"category": "support", "confidence": 0.9})
    provider = FakeProvider([classification, "support reply"])

    @agent(model="fake-model", provider=provider)
    def classify(ticket: str) -> dict:
        """Classify."""

    @agent(model="fake-model", provider=provider)
    def handle_billing(ticket: str) -> str:
        """Handle billing."""

    @agent(model="fake-model", provider=provider)
    def handle_support(ticket: str) -> str:
        """Handle support."""

    triage = pipeline(classify >> branch(billing=handle_billing, support=handle_support))
    result = run(triage, ticket="my app crashed")

    assert result.output == "support reply"


def test_routes_on_pydantic_model_output_using_category_field():
    import json

    classification = json.dumps({"category": "billing", "confidence": 0.8})
    provider = FakeProvider([classification, "billing reply"])

    @agent(model="fake-model", provider=provider)
    def classify(ticket: str) -> Classification:
        """Classify."""

    @agent(model="fake-model", provider=provider)
    def handle_billing(ticket: str) -> str:
        """Handle billing."""

    @agent(model="fake-model", provider=provider)
    def handle_support(ticket: str) -> str:
        """Handle support."""

    triage = pipeline(classify >> branch(billing=handle_billing, support=handle_support))
    result = run(triage, ticket="charge me twice")

    assert result.output == "billing reply"


def test_falls_back_to_default_on_unmatched_key():
    provider = FakeProvider(["unknown_category", "general reply"])

    @agent(model="fake-model", provider=provider)
    def classify(ticket: str) -> str:
        """Classify."""

    @agent(model="fake-model", provider=provider)
    def handle_billing(ticket: str) -> str:
        """Handle billing."""

    @agent(model="fake-model", provider=provider)
    def handle_general(ticket: str) -> str:
        """Handle general."""

    triage = pipeline(classify >> branch(billing=handle_billing, default=handle_general))
    result = run(triage, ticket="something weird")

    assert result.output == "general reply"


def test_raises_branch_error_on_unmatched_key_with_no_default():
    provider = FakeProvider(["unknown_category"])

    @agent(model="fake-model", provider=provider)
    def classify(ticket: str) -> str:
        """Classify."""

    @agent(model="fake-model", provider=provider)
    def handle_billing(ticket: str) -> str:
        """Handle billing."""

    triage = pipeline(classify >> branch(billing=handle_billing))

    with pytest.raises(BranchError) as exc_info:
        run(triage, ticket="something weird")
    assert exc_info.value.key == "unknown_category"


def test_run_branch_directly_without_pipeline_wrapper():
    provider = FakeProvider(["handled directly"])

    @agent(model="fake-model", provider=provider)
    def unreachable(ticket: str) -> str:
        """Never matches — there's no upstream classifier output to route on."""

    @agent(model="fake-model", provider=provider)
    def handle_it(ticket: str) -> str:
        """Handle it."""

    # No upstream classifier — branch() run standalone has no prior result to extract
    # a routing key from, so it always falls through to default.
    result = run(branch(billing=unreachable, default=handle_it), ticket="anything")
    assert result.output == "handled directly"
