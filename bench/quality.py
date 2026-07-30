"""Semantic retrieval quality: recall@k on real, meaningful text.

This is deliberately different from bench/run.py, which uses synthetic
random vectors to isolate storage/ANN engine performance — that
methodology is explicitly incapable of answering "which embedding model
actually understands text better," because random vectors carry no
semantic content to understand. This script measures exactly that:
given a query and a handful of topically-similar passages, does the
embedder rank the actually-correct passage above the near-miss
distractors?

**Honestly scoped:** this is a small (~18 examples), hand-curated eval
spanning the shapes of content rmbr actually indexes — remembered
preferences, documentation, code — not a large-scale published IR
benchmark (BEIR, MTEB, etc.). It's sized to catch a real, meaningful
quality difference on rmbr's actual use case, not to be a definitive
academic result. Every example is checked in below in plain sight, not
hidden in a data file, so the methodology is auditable at a glance.

    python bench/quality.py                    # benches the local default
    OPENAI_API_KEY=sk-... python bench/quality.py --with-openai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402

from rmbr.embed import Embedder, FastEmbedEmbedder  # noqa: E402

# Each example: a query, the one correct passage, and several distractors
# that share the same topic area but aren't the right answer - a
# retrieval test only means something if the wrong answers are plausible.
EXAMPLES: list[dict] = [
    # -- memory-style: short first-person facts an agent might remember --
    {
        "category": "memory",
        "query": "what color scheme does the user like?",
        "correct": "user prefers dark mode and short answers",
        "distractors": [
            "user prefers light mode for accessibility reasons",
            "user's favorite color is blue",
            "user wants the sidebar collapsed by default",
        ],
    },
    {
        "category": "memory",
        "query": "what timezone should I use for scheduling?",
        "correct": "user's timezone is PST, please schedule meetings accordingly",
        "distractors": [
            "user is usually available in the mornings",
            "user's timezone is EST for billing purposes",
            "user prefers async communication over meetings",
        ],
    },
    {
        "category": "memory",
        "query": "how does the user want errors reported?",
        "correct": "user wants error messages to include the full stack trace, not a summary",
        "distractors": [
            "user wants error notifications sent via email",
            "user prefers warnings to be silent unless critical",
            "user wants a daily digest of all logged errors",
        ],
    },
    {
        "category": "memory",
        "query": "what's the user's preferred language for scripts?",
        "correct": "user prefers Python for automation scripts over bash",
        "distractors": [
            "user prefers TypeScript for frontend work",
            "user avoids using Python for performance-critical code",
            "user's team standardized on Go for backend services",
        ],
    },
    {
        "category": "memory",
        "query": "does the user want confirmation before destructive actions?",
        "correct": "user wants to be asked for confirmation before any file deletion",
        "distractors": [
            "user wants automatic backups before any file edit",
            "user disabled confirmation prompts for git commits",
            "user prefers deletions to be reversible via trash, not confirmed",
        ],
    },
    # -- documentation-style: FAQ / how-to passages --
    {
        "category": "docs",
        "query": "how do I deploy the app to production?",
        "correct": "To deploy, run ./deploy.sh from the repo root after setting the API_KEY environment variable.",
        "distractors": [
            "To run tests locally, use pytest tests/ from the repo root.",
            "To roll back a deployment, use ./rollback.sh with the target version tag.",
            "To set up a local dev environment, run ./setup.sh and follow the prompts.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I reset my password?",
        "correct": "Click 'Forgot password' on the login page and follow the emailed reset link, valid for 24 hours.",
        "distractors": [
            "To change your email address, go to Account Settings and verify the new address.",
            "To enable two-factor authentication, go to Security Settings and scan the QR code.",
            "If your account is locked, contact support with your account ID.",
        ],
    },
    {
        "category": "docs",
        "query": "what ports does the service listen on?",
        "correct": "The API server listens on port 8000, and the admin dashboard listens on port 8080.",
        "distractors": [
            "The database listens on port 5432 by default.",
            "The message queue broker uses port 5672 for AMQP connections.",
            "The health check endpoint is available at /healthz on the API server.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I configure rate limiting?",
        "correct": "Set RATE_LIMIT_PER_MINUTE in the config file to control how many requests each API key can make.",
        "distractors": [
            "Set LOG_LEVEL to DEBUG for verbose request logging.",
            "Set MAX_UPLOAD_SIZE_MB to control the maximum file upload size.",
            "Set SESSION_TIMEOUT_MINUTES to control how long a login session stays valid.",
        ],
    },
    {
        "category": "docs",
        "query": "how do backups work?",
        "correct": "Backups run nightly at 2am UTC and are retained for 30 days in cold storage.",
        "distractors": [
            "Log files are rotated daily and retained for 7 days.",
            "Cache entries expire after 15 minutes of inactivity.",
            "Metrics are aggregated hourly and retained for 90 days.",
        ],
    },
    {
        "category": "docs",
        "query": "what happens if the primary database goes down?",
        "correct": "The system automatically fails over to the read replica within 30 seconds and pages the on-call engineer.",
        "distractors": [
            "The system queues writes locally if the network connection drops.",
            "The system retries failed API calls up to 3 times with exponential backoff.",
            "The system logs a warning if disk usage exceeds 90%.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I add a new team member?",
        "correct": "Go to Team Settings, click Invite, and enter their email - they'll get an invite link valid for 7 days.",
        "distractors": [
            "Go to Billing to upgrade your plan for more seats.",
            "Go to Integrations to connect a new third-party tool.",
            "Go to API Keys to generate a new key for a service account.",
        ],
    },
    # -- code-style: function purpose from a docstring or signature --
    {
        "category": "code",
        "query": "which function validates user input?",
        "correct": "def validate_email(address: str) -> bool: checks the address against a regex pattern and returns whether it's a valid email format.",
        "distractors": [
            "def send_email(to: str, subject: str, body: str) -> None: sends an email via the configured SMTP server.",
            "def hash_password(password: str) -> str: hashes a password using bcrypt before storage.",
            "def format_currency(amount: float) -> str: formats a float as a currency string with two decimal places.",
        ],
    },
    {
        "category": "code",
        "query": "how does the retry logic work?",
        "correct": "def retry_with_backoff(fn, max_attempts=3): retries a function call with exponential backoff, raising the last exception if all attempts fail.",
        "distractors": [
            "def cache_result(fn): memoizes a function's return value based on its arguments.",
            "def rate_limit(fn, calls_per_second): wraps a function to enforce a maximum call rate.",
            "def log_execution_time(fn): wraps a function to log how long it took to run.",
        ],
    },
    {
        "category": "code",
        "query": "which function handles pagination?",
        "correct": "def paginate(query, page: int, page_size: int = 20): applies LIMIT/OFFSET to a query based on the requested page number.",
        "distractors": [
            "def sort_results(items, key, reverse=False): sorts a list of items by the given key.",
            "def filter_active(items): returns only the items whose status is 'active'.",
            "def deduplicate(items, key): removes duplicate items from a list based on a key function.",
        ],
    },
    {
        "category": "code",
        "query": "how is the auth token verified?",
        "correct": "def verify_token(token: str) -> dict: decodes and verifies a JWT, raising an error if it's expired or has an invalid signature.",
        "distractors": [
            "def generate_token(user_id: str) -> str: creates a new signed JWT for a given user id.",
            "def refresh_session(session_id: str) -> None: extends a session's expiry time.",
            "def revoke_all_tokens(user_id: str) -> None: invalidates every active token for a user.",
        ],
    },
    {
        "category": "code",
        "query": "which function connects to the database?",
        "correct": "def get_connection() -> Connection: opens a new connection to the configured Postgres database, reusing a pooled connection if available.",
        "distractors": [
            "def run_migrations() -> None: applies any pending database schema migrations.",
            "def seed_test_data() -> None: inserts fixture data into the database for testing.",
            "def backup_database(path: str) -> None: dumps the database to a file at the given path.",
        ],
    },
    {
        "category": "code",
        "query": "how are search results ranked?",
        "correct": "def rank_results(query, candidates): scores each candidate by relevance to the query and returns them sorted highest-first.",
        "distractors": [
            "def dedupe_results(candidates): removes near-duplicate results from a candidate list.",
            "def highlight_matches(text, query): wraps matching substrings in a result's text with highlight markers.",
            "def paginate_results(candidates, page): slices a candidate list into pages for display.",
        ],
    },
]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def evaluate(embedder: Embedder, examples: list[dict], k: int = 1) -> dict:
    """Returns {"recall_at_k": float, "by_category": {...}, "misses": [...]}."""
    hits = 0
    by_category: dict[str, list[int]] = {}
    misses = []

    for ex in examples:
        passages = [ex["correct"]] + ex["distractors"]
        query_vector = embedder.embed([ex["query"]])[0]
        passage_vectors = embedder.embed(passages)
        scores = [_cosine(query_vector, v) for v in passage_vectors]
        ranked = sorted(range(len(passages)), key=lambda i: -scores[i])
        correct_rank = ranked.index(0)  # index 0 is always the correct passage
        hit = correct_rank < k
        hits += hit
        by_category.setdefault(ex["category"], []).append(hit)
        if not hit:
            misses.append({"query": ex["query"], "correct": ex["correct"], "rank": correct_rank})

    return {
        "recall_at_k": hits / len(examples),
        "by_category": {cat: sum(v) / len(v) for cat, v in by_category.items()},
        "misses": misses,
    }


def _print_report(name: str, result: dict) -> None:
    print(f"\n{name}")
    print(f"  overall recall@1: {result['recall_at_k']:.3f} ({len(EXAMPLES)} examples)")
    for cat, score in result["by_category"].items():
        print(f"    {cat:<10} recall@1: {score:.3f}")
    if result["misses"]:
        print("  missed:")
        for miss in result["misses"]:
            print(f"    - {miss['query']!r} -> correct passage ranked #{miss['rank'] + 1}, not #1")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--with-openai", action="store_true", help="also evaluate OpenAIEmbedder (needs OPENAI_API_KEY)"
    )
    args = parser.parse_args()

    print(f"Evaluating on {len(EXAMPLES)} hand-curated (query, correct, distractors) examples...")

    default_embedder = FastEmbedEmbedder()
    default_result = evaluate(default_embedder, EXAMPLES)
    _print_report(f"rmbr default ({default_embedder.model_name})", default_result)

    if args.with_openai:
        from rmbr.embed import OpenAIEmbedder

        openai_embedder = OpenAIEmbedder()
        openai_result = evaluate(openai_embedder, EXAMPLES)
        _print_report(f"OpenAIEmbedder ({openai_embedder.model_name})", openai_result)


if __name__ == "__main__":
    main()
