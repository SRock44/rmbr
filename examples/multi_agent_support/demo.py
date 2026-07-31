"""Multi-agent customer support over one shared rmbr .db file.

Three agents share `support.db`:

  - "billing" and "technical" are specialist agents. Each has its own
    namespace-scoped Memory (customer-specific notes) and Index (product
    docs). Neither can read the other's namespace - not because a prompt
    tells them not to, but because rmbr's Policy denies it by default,
    and the tools handed to the LLM (idx.as_tool(), mem.as_tools()) don't
    expose a namespace parameter for a prompt injection to override. The
    only way to read another namespace is the raw `namespaces=` kwarg on
    `recall()`/`search()` - which never reaches the model, because it's
    not in any tool schema.

  - "supervisor" is granted read access to every namespace (`read="*"`),
    for escalation and end-of-shift review - the one agent for whom that
    kwarg *is* used, directly in trusted code, never via the LLM.

rmbr never calls an LLM itself (see rmbr.Policy's docstring) - Claude is
this script's choice, wired in manually below.

Run:
    pip install rmbr anthropic
    export ANTHROPIC_API_KEY=...
    python demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic

from rmbr import Index, Memory, Policy

DB_PATH = Path(__file__).parent / "support.db"
MODEL = "claude-opus-5"

BILLING_DOCS = [
    "Refunds are issued to the original payment method within 5-7 business days of approval. "
    "Orders over 90 days old require a manager override.",
    "Customers can cancel a subscription anytime from Account > Subscription > Cancel. "
    "Cancellation takes effect at the end of the current billing period; no partial-month refunds.",
    "A duplicate charge is almost always a temporary authorization hold, not a second real charge - "
    "holds drop off within 3-5 business days. If a second charge posts (not just a hold) after 5 "
    "days, issue a manual refund.",
]

TECHNICAL_DOCS = [
    "A login redirect loop is caused by a stale session cookie. Ask the customer to clear cookies "
    "for the site or try an incognito window; this resolves about 90% of cases.",
    "Cross-device sync can lag up to 15 minutes under normal load. If it's been longer, check "
    "Settings > Sync Status for an error banner before escalating.",
    "CSV export fails silently for accounts with more than 50,000 rows. Recommend the customer use "
    "the API export endpoint instead, which has no row limit.",
]

SPECIALIST_SYSTEM_PROMPTS = {
    "billing": (
        "You are a billing support agent for TechCorp. Use search_kb to check policy before "
        "answering. Use recall to check for prior notes on this customer before answering. Use "
        "remember to save any customer-specific fact a future billing conversation would need "
        "(e.g. a refund already issued, a cancellation already processed). Be concise - 2-3 "
        "sentences."
    ),
    "technical": (
        "You are a technical support agent for TechCorp. Use search_kb to check known issues "
        "before answering. Use recall to check prior notes on this customer. Use remember to save "
        "any customer-specific fact a future conversation would need. Be concise - 2-3 sentences."
    ),
}


def seed_knowledge_base(policy: Policy) -> None:
    for namespace, docs in (("billing", BILLING_DOCS), ("technical", TECHNICAL_DOCS)):
        idx = Index(str(DB_PATH), namespace=namespace, policy=policy)
        for doc in docs:
            idx.add_text(doc)
        idx.close()


def run_specialist(namespace: str, policy: Policy, customer_id: str, message: str) -> str:
    """One specialist agent's turn: tools scoped to its own namespace only."""
    client = anthropic.Anthropic()
    idx = Index(str(DB_PATH), namespace=namespace, policy=policy)
    mem = Memory(str(DB_PATH), namespace, policy=policy)

    tools_by_name = {t.name: t for t in [idx.as_tool(name="search_kb"), *mem.as_tools()]}
    tool_defs = [t.to_anthropic() for t in tools_by_name.values()]

    messages = [{"role": "user", "content": f"Customer {customer_id} says: {message}"}]
    final_text = ""

    for _ in range(5):  # bounded agentic loop
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SPECIALIST_SYSTEM_PROMPTS[namespace],
            tools=tool_defs,
            messages=messages,
        )

        if response.stop_reason == "refusal":
            final_text = "(request declined by safety classifiers)"
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if not tool_uses:
            final_text = next((b.text for b in response.content if b.type == "text"), "")
            break

        tool_results = []
        for block in tool_uses:
            tool = tools_by_name[block.name]
            print(f"    [{namespace}] -> {block.name}({block.input})")
            result = tool.call(**block.input)
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
            )
        messages.append({"role": "user", "content": tool_results})

    idx.close()
    mem.close()
    return final_text


def route(message: str) -> str:
    """Supervisor's turn: classify the message with a forced tool call - deterministic
    dispatch code, the model only supplies the classification."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=64,
        tools=[
            {
                "name": "route_to_specialist",
                "description": "Route a customer message to the right specialist team.",
                "input_schema": {
                    "type": "object",
                    "properties": {"team": {"type": "string", "enum": ["billing", "technical"]}},
                    "required": ["team"],
                },
            }
        ],
        tool_choice={"type": "tool", "name": "route_to_specialist"},
        messages=[{"role": "user", "content": message}],
    )
    tool_use = next(b for b in response.content if b.type == "tool_use")
    return tool_use.input["team"]


def demonstrate_isolation(policy: Policy) -> None:
    """The technical agent tries to read the billing agent's memory directly - not
    through the LLM (that path doesn't even expose a namespace parameter, see
    SPECIALIST_SYSTEM_PROMPTS above), but as a raw API call, to show the denial is
    enforced by Policy itself, not by the model choosing to behave."""
    technical_mem = Memory(str(DB_PATH), "technical", policy=policy)
    print("\n--- Policy enforcement check ---")
    print("technical agent attempting mem.recall(..., namespaces='billing'):")
    try:
        technical_mem.recall("refund", namespaces="billing")
        print("  UNEXPECTED: read succeeded")
    except PermissionError as e:
        print(f"  denied, as expected: {e}")
    technical_mem.close()


def supervisor_audit(policy: Policy) -> None:
    """The supervisor - and only the supervisor - can read across both namespaces."""
    print("\n--- Supervisor end-of-shift audit (reads across every namespace) ---")
    supervisor_mem = Memory(str(DB_PATH), "supervisor", policy=policy)
    hits = supervisor_mem.recall("customer issue", namespaces="*", k=10)
    for hit in hits:
        print(f"  [{hit.namespace}] {hit.text}")
    supervisor_mem.close()


def main() -> None:
    DB_PATH.unlink(missing_ok=True)

    policy = Policy()  # deny-by-default
    policy.allow("supervisor", read="*")

    print("Seeding billing + technical knowledge bases...")
    seed_knowledge_base(policy)

    conversations = [
        ("cust_482", "I was charged twice for my subscription this month, can I get one refund back?"),
        ("cust_119", "I keep getting bounced back to the login page in a loop, nothing works."),
    ]

    for customer_id, message in conversations:
        print(f"\n=== {customer_id}: {message!r} ===")
        team = route(message)
        print(f"  routed to: {team}")
        answer = run_specialist(team, policy, customer_id, message)
        print(f"  [{team} agent] {answer}")

    demonstrate_isolation(policy)
    supervisor_audit(policy)


if __name__ == "__main__":
    main()
