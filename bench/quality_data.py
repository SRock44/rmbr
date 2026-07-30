"""Eval data for bench/quality.py - 150 hand-written (query, correct
passage, distractors) examples, 50 per category (memory/docs/code),
organized into 10 sub-themes of 5 examples each per category for real
topical diversity rather than repetitive near-duplicates.

Distractors are deliberately plausible near-misses within the same
sub-theme, not random unrelated text - a retrieval eval only means
something if the wrong answers could trick a weak embedder.
"""

from __future__ import annotations

EXAMPLES: list[dict] = [
    # ============================== MEMORY ==============================
    # -- UI / appearance --
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
        "query": "what font size should the interface use?",
        "correct": "user has low vision and needs the UI font size set to at least 18px",
        "distractors": [
            "user prefers a monospace font for code blocks",
            "user wants line spacing increased for readability",
            "user disabled animations due to motion sensitivity",
        ],
    },
    {
        "category": "memory",
        "query": "does the user want a compact or spacious layout?",
        "correct": "user prefers a compact, information-dense layout over spacious cards",
        "distractors": [
            "user wants the navigation bar pinned to the top",
            "user prefers tabs over a single scrolling page",
            "user wants breadcrumbs shown on every page",
        ],
    },
    {
        "category": "memory",
        "query": "what theme did the user choose for the code editor?",
        "correct": "user's code editor theme is set to 'Solarized Dark'",
        "distractors": [
            "user's terminal color scheme is the default",
            "user disabled syntax highlighting for markdown files",
            "user prefers 2-space indentation over tabs",
        ],
    },
    {
        "category": "memory",
        "query": "how does the user want notifications displayed?",
        "correct": "user wants notifications to appear as a small badge, not a popup",
        "distractors": [
            "user wants email digests instead of push notifications",
            "user muted notifications during work hours",
            "user wants a sound alert for urgent messages only",
        ],
    },
    # -- communication style --
    {
        "category": "memory",
        "query": "how verbose does the user want responses to be?",
        "correct": "user prefers short, direct answers without preamble",
        "distractors": [
            "user wants step-by-step explanations for complex topics",
            "user asked to always include code examples",
            "user wants a summary at the top of long responses",
        ],
    },
    {
        "category": "memory",
        "query": "what tone does the user prefer?",
        "correct": "user prefers a casual, informal tone over formal business language",
        "distractors": [
            "user wants technical jargon avoided when possible",
            "user prefers being addressed by first name",
            "user dislikes excessive enthusiasm or exclamation points",
        ],
    },
    {
        "category": "memory",
        "query": "should follow-up questions be asked?",
        "correct": "user prefers the assistant make a reasonable assumption rather than ask clarifying questions",
        "distractors": [
            "user wants confirmation before any irreversible action",
            "user prefers being shown multiple options to choose from",
            "user wants a rationale given for every recommendation",
        ],
    },
    {
        "category": "memory",
        "query": "how does the user want code changes explained?",
        "correct": "user wants a one-line summary of what changed, not a full diff walkthrough",
        "distractors": [
            "user wants every function documented with docstrings",
            "user prefers commit messages under 50 characters",
            "user wants tests written before implementation code",
        ],
    },
    {
        "category": "memory",
        "query": "what format does the user want for lists?",
        "correct": "user prefers numbered lists over bullet points for sequential steps",
        "distractors": [
            "user wants tables used instead of prose when comparing options",
            "user prefers headers over bold text for section titles",
            "user dislikes emoji in professional documents",
        ],
    },
    # -- scheduling / availability --
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
        "query": "when is the user typically available for calls?",
        "correct": "user is available for calls between 10am and 2pm on weekdays",
        "distractors": [
            "user works a four-day week, off on Fridays",
            "user prefers meetings kept under 30 minutes",
            "user is traveling next week with limited availability",
        ],
    },
    {
        "category": "memory",
        "query": "how far in advance does the user want meeting reminders?",
        "correct": "user wants a reminder 15 minutes before any scheduled meeting",
        "distractors": [
            "user wants a daily agenda summary each morning",
            "user prefers meetings booked at least 24 hours in advance",
            "user wants recurring meetings reviewed monthly",
        ],
    },
    {
        "category": "memory",
        "query": "does the user take a lunch break?",
        "correct": "user blocks 12pm to 1pm daily for lunch and doesn't want it scheduled over",
        "distractors": [
            "user prefers back-to-back meetings without buffer time",
            "user works remotely three days a week",
            "user's standup meeting is at 9:15am daily",
        ],
    },
    {
        "category": "memory",
        "query": "what's the user's policy on weekend work?",
        "correct": "user does not want to be contacted about work matters on weekends",
        "distractors": [
            "user checks email once daily on vacation",
            "user prefers deadlines set for Friday, not Monday",
            "user takes public holidays off according to the US calendar",
        ],
    },
    # -- tools / languages --
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
        "query": "which package manager does the user use?",
        "correct": "user's team uses pnpm, not npm or yarn, for all JavaScript projects",
        "distractors": [
            "user prefers pip over conda for Python environments",
            "user's Docker images are built with buildx for multi-arch support",
            "user's CI pipeline caches dependencies between runs",
        ],
    },
    {
        "category": "memory",
        "query": "what testing framework does the user prefer?",
        "correct": "user's team writes tests with pytest, not unittest",
        "distractors": [
            "user prefers integration tests over heavy mocking",
            "user's coverage threshold is set to 80 percent",
            "user runs tests in parallel using pytest-xdist",
        ],
    },
    {
        "category": "memory",
        "query": "which cloud provider does the user's team use?",
        "correct": "user's infrastructure runs on AWS, specifically the us-east-1 region",
        "distractors": [
            "user's static assets are served from a CDN",
            "user's team evaluated GCP but decided against migrating",
            "user's database backups are stored in a separate account",
        ],
    },
    {
        "category": "memory",
        "query": "what's the user's git branching convention?",
        "correct": "user's team uses trunk-based development with short-lived feature branches",
        "distractors": [
            "user squashes commits before merging to main",
            "user requires two approvals before merging a PR",
            "user's release branches are cut weekly",
        ],
    },
    # -- workflow / confirmation --
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
    {
        "category": "memory",
        "query": "does the user want changes auto-saved?",
        "correct": "user wants documents auto-saved every 30 seconds while editing",
        "distractors": [
            "user prefers manual save with a keyboard shortcut",
            "user wants version history kept for 90 days",
            "user disabled auto-formatting on save",
        ],
    },
    {
        "category": "memory",
        "query": "how does the user want pull requests reviewed?",
        "correct": "user wants at least one reviewer to approve before merging any PR",
        "distractors": [
            "user prefers PRs kept under 400 lines of changes",
            "user wants CI to pass before a PR can be reviewed",
            "user squashes commits when merging",
        ],
    },
    {
        "category": "memory",
        "query": "what's the user's policy on force-pushing?",
        "correct": "user never wants force-push used on the main branch under any circumstance",
        "distractors": [
            "user rebases feature branches before opening a PR",
            "user prefers merge commits over rebase for team branches",
            "user tags releases immediately after merging to main",
        ],
    },
    {
        "category": "memory",
        "query": "does the user want dependencies auto-updated?",
        "correct": "user wants dependency updates reviewed manually, not applied automatically",
        "distractors": [
            "user runs a security audit on dependencies monthly",
            "user pins exact versions in the lockfile",
            "user's team avoids pre-release package versions",
        ],
    },
    # -- personal / role facts --
    {
        "category": "memory",
        "query": "what is the user's role on the team?",
        "correct": "user is the lead backend engineer responsible for the API service",
        "distractors": [
            "user recently joined the team last quarter",
            "user manages the on-call rotation schedule",
            "user is the point of contact for database questions",
        ],
    },
    {
        "category": "memory",
        "query": "which project is the user currently focused on?",
        "correct": "user is currently focused on migrating the billing system to a new provider",
        "distractors": [
            "user previously worked on the notifications service",
            "user's next project starts after the current sprint ends",
            "user is blocked waiting on a design review",
        ],
    },
    {
        "category": "memory",
        "query": "who does the user report to?",
        "correct": "user reports directly to the VP of engineering",
        "distractors": [
            "user's manager is out on leave until next month",
            "user has a 1:1 scheduled every other Tuesday",
            "user mentors two junior engineers on the team",
        ],
    },
    {
        "category": "memory",
        "query": "what company does the user work for?",
        "correct": "user works at a mid-size fintech startup with about 80 employees",
        "distractors": [
            "user previously worked at a large enterprise software company",
            "user's company is fully remote with no physical office",
            "user's company just closed a Series B funding round",
        ],
    },
    {
        "category": "memory",
        "query": "what's the user's area of technical expertise?",
        "correct": "user specializes in distributed systems and database internals",
        "distractors": [
            "user has beginner-level experience with machine learning",
            "user is learning Rust in their spare time",
            "user previously worked as a frontend developer",
        ],
    },
    # -- security / privacy --
    {
        "category": "memory",
        "query": "how does the user want secrets stored?",
        "correct": "user wants secrets stored in a vault, never committed to the repo",
        "distractors": [
            "user rotates API keys every 90 days",
            "user requires two-factor authentication for all team accounts",
            "user wants access logs reviewed weekly",
        ],
    },
    {
        "category": "memory",
        "query": "what's the user's policy on sharing data with third parties?",
        "correct": "user does not want customer data shared with any third-party analytics tool",
        "distractors": [
            "user requires a data processing agreement before onboarding a vendor",
            "user's data retention policy is 2 years for logs",
            "user wants PII redacted from support tickets",
        ],
    },
    {
        "category": "memory",
        "query": "does the user want audit logging enabled?",
        "correct": "user wants every admin action logged with a timestamp and actor",
        "distractors": [
            "user wants failed login attempts rate-limited",
            "user requires SSO for all internal tools",
            "user's team does quarterly access reviews",
        ],
    },
    {
        "category": "memory",
        "query": "what's the user's stance on using AI tools with sensitive code?",
        "correct": "user does not want proprietary source code sent to third-party AI services",
        "distractors": [
            "user is comfortable using AI tools for documentation writing",
            "user wants AI-generated code reviewed before merging",
            "user's company has an approved list of AI tools",
        ],
    },
    {
        "category": "memory",
        "query": "how does the user want backups encrypted?",
        "correct": "user wants all backups encrypted at rest using AES-256",
        "distractors": [
            "user wants backups tested with a restore drill quarterly",
            "user's backup retention window is 30 days",
            "user wants backup failures to page the on-call engineer",
        ],
    },
    # -- error handling / logging --
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
        "query": "what log level does the user want in production?",
        "correct": "user wants production services running at INFO log level, not DEBUG",
        "distractors": [
            "user wants logs shipped to a centralized log aggregator",
            "user's staging environment runs at DEBUG level",
            "user wants structured JSON logs, not plain text",
        ],
    },
    {
        "category": "memory",
        "query": "how should retries be handled on failure?",
        "correct": "user wants failed background jobs retried up to 3 times before alerting",
        "distractors": [
            "user wants a circuit breaker on the payment gateway integration",
            "user prefers idempotent job handlers",
            "user wants dead-letter queues for jobs that exhaust retries",
        ],
    },
    {
        "category": "memory",
        "query": "what's the user's alerting threshold for latency?",
        "correct": "user wants an alert fired if p95 latency exceeds 500ms for 5 minutes",
        "distractors": [
            "user wants a weekly performance report generated automatically",
            "user's error budget is 99.9 percent uptime per month",
            "user wants alerts routed to a dedicated Slack channel",
        ],
    },
    {
        "category": "memory",
        "query": "does the user want warnings treated as errors?",
        "correct": "user wants the build to fail on any compiler warning, not just errors",
        "distractors": [
            "user wants linting run automatically on every commit",
            "user's CI pipeline runs type checking as a separate step",
            "user wants deprecation warnings tracked in a backlog",
        ],
    },
    # -- integrations --
    {
        "category": "memory",
        "query": "which chat tool does the user want notifications sent to?",
        "correct": "user wants deployment notifications sent to the #eng-deploys Slack channel",
        "distractors": [
            "user's team uses Discord for informal discussion",
            "user wants a weekly summary emailed instead of real-time pings",
            "user muted the general channel but kept alerts on",
        ],
    },
    {
        "category": "memory",
        "query": "which calendar does the user use?",
        "correct": "user's calendar is Google Calendar, synced with their work email",
        "distractors": [
            "user blocks focus time every morning",
            "user shares their calendar with their direct reports",
            "user's timezone is set incorrectly and needs fixing",
        ],
    },
    {
        "category": "memory",
        "query": "what issue tracker does the user's team use?",
        "correct": "user's team tracks work in Linear, not Jira or GitHub Issues",
        "distractors": [
            "user prefers issues linked to the relevant PR automatically",
            "user's sprint length is two weeks",
            "user wants a burndown chart reviewed at each retro",
        ],
    },
    {
        "category": "memory",
        "query": "which payment processor does the user's product use?",
        "correct": "user's product processes payments through Stripe",
        "distractors": [
            "user is evaluating adding PayPal as a second option",
            "user's refund policy allows 30 days",
            "user's billing runs on a monthly subscription cycle",
        ],
    },
    {
        "category": "memory",
        "query": "what analytics tool does the user use?",
        "correct": "user tracks product usage with PostHog, self-hosted",
        "distractors": [
            "user wants a monthly analytics report emailed to stakeholders",
            "user's marketing site uses a separate analytics tool",
            "user opted out of sending anonymous usage data",
        ],
    },
    # -- output formatting --
    {
        "category": "memory",
        "query": "what date format does the user prefer?",
        "correct": "user prefers dates written as YYYY-MM-DD, not MM/DD/YYYY",
        "distractors": [
            "user wants times shown in 24-hour format",
            "user's reports are generated in the user's local timezone",
            "user wants relative dates like 'yesterday' for recent events",
        ],
    },
    {
        "category": "memory",
        "query": "what currency should amounts be shown in?",
        "correct": "user wants all monetary amounts displayed in USD, not converted",
        "distractors": [
            "user's invoices are generated at the end of each month",
            "user wants tax calculated based on billing address",
            "user's pricing page shows amounts rounded to the nearest dollar",
        ],
    },
    {
        "category": "memory",
        "query": "how does the user want large numbers formatted?",
        "correct": "user wants large numbers formatted with commas, like 1,000,000",
        "distractors": [
            "user wants percentages shown with one decimal place",
            "user prefers scientific notation avoided in reports",
            "user wants negative numbers shown in parentheses, not with a minus sign",
        ],
    },
    {
        "category": "memory",
        "query": "what unit system does the user prefer?",
        "correct": "user prefers metric units, not imperial, for all measurements",
        "distractors": [
            "user wants file sizes shown in binary units like MiB, not MB",
            "user's shipping calculator defaults to the destination country's units",
            "user wants temperature shown in both Celsius and Fahrenheit",
        ],
    },
    {
        "category": "memory",
        "query": "how should the user's name be displayed?",
        "correct": "user wants to be addressed by their preferred name, not their legal name",
        "distractors": [
            "user's email signature includes their pronouns",
            "user's display name in the app is different from their username",
            "user wants their title included in formal correspondence",
        ],
    },
    # ============================== DOCS ==============================
    # -- deployment --
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
        "query": "how do I roll back a bad deployment?",
        "correct": "Run ./rollback.sh with the version tag you want to restore; it takes about 2 minutes to complete.",
        "distractors": [
            "To deploy a new version, push a tag matching v*.*.* and CI handles the rest.",
            "To view deployment history, check the Releases tab in the admin dashboard.",
            "To pause auto-deploys, toggle the freeze flag in the deploy config.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I deploy to a staging environment?",
        "correct": "Push to the 'staging' branch and the staging environment redeploys automatically within 5 minutes.",
        "distractors": [
            "Production deploys require a manual approval step in the CI pipeline.",
            "Staging uses a separate database that gets reset nightly.",
            "Feature flags let you test changes in production without a full deploy.",
        ],
    },
    {
        "category": "docs",
        "query": "what happens during a blue-green deployment?",
        "correct": "Traffic is gradually shifted from the old version to the new version over 10 minutes, with automatic rollback if error rates spike.",
        "distractors": [
            "Canary deployments route 5 percent of traffic to the new version first.",
            "Deployment logs are retained for 30 days for auditing.",
            "Each deployment gets a unique build ID shown in the dashboard.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I check if a deployment succeeded?",
        "correct": "Check the /healthz endpoint returns 200, and confirm the version shown matches what you deployed.",
        "distractors": [
            "Deployment notifications are sent to the #eng-deploys channel.",
            "A failed deployment automatically triggers a rollback after 3 failed health checks.",
            "You can view real-time deployment progress in the CI dashboard.",
        ],
    },
    # -- account / auth --
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
        "query": "how do I enable two-factor authentication?",
        "correct": "Go to Security Settings, click Enable 2FA, and scan the QR code with an authenticator app.",
        "distractors": [
            "To reset your password, use the link on the login page.",
            "Session tokens expire after 30 days of inactivity.",
            "You can view your active login sessions under Security Settings.",
        ],
    },
    {
        "category": "docs",
        "query": "how long does a login session last?",
        "correct": "Login sessions stay active for 30 days unless you explicitly log out or change your password.",
        "distractors": [
            "API tokens expire after 90 days and must be regenerated.",
            "Failed login attempts lock the account after 5 tries for 15 minutes.",
            "You can revoke a specific session from the Security Settings page.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I connect single sign-on?",
        "correct": "Admins can configure SSO under Organization Settings by entering your identity provider's SAML metadata URL.",
        "distractors": [
            "Individual users can't enable SSO themselves - it's an org-wide setting.",
            "SSO-enabled accounts can't also use a password to log in.",
            "SCIM provisioning syncs users automatically once SSO is configured.",
        ],
    },
    {
        "category": "docs",
        "query": "what happens if I get locked out of my account?",
        "correct": "If your account is locked after too many failed attempts, contact support with your account ID to unlock it.",
        "distractors": [
            "Password reset links expire after 24 hours for security.",
            "You can set up a backup email address in Account Settings.",
            "Two-factor recovery codes are shown once when you enable 2FA - save them.",
        ],
    },
    # -- API / config --
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
        "query": "how do I authenticate API requests?",
        "correct": "Include your API key in the Authorization header as 'Bearer <key>' on every request.",
        "distractors": [
            "API keys can be scoped to read-only or read-write permissions.",
            "Rate limit headers are returned on every response showing your remaining quota.",
            "You can generate multiple API keys per account for different services.",
        ],
    },
    {
        "category": "docs",
        "query": "what's the maximum request body size?",
        "correct": "The API rejects request bodies larger than 10MB with a 413 status code.",
        "distractors": [
            "File uploads are limited to 100MB per file.",
            "Batch endpoints accept up to 500 items per request.",
            "Response payloads are gzip-compressed by default.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I paginate through API results?",
        "correct": "Use the 'cursor' query parameter from the previous response to fetch the next page of results.",
        "distractors": [
            "Results are sorted by creation date descending by default.",
            "The 'limit' parameter caps the page size at 100 items.",
            "Filtering is supported via query parameters like status and created_after.",
        ],
    },
    # -- troubleshooting / incidents --
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
        "query": "what do I do if a deployment causes an outage?",
        "correct": "Immediately run ./rollback.sh to the last known-good version, then open an incident in the status page tool.",
        "distractors": [
            "Post-incident reviews are scheduled within 48 hours of resolution.",
            "The on-call rotation is managed in the incident response tool.",
            "Status page updates should be posted every 30 minutes during an active incident.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I check current system status?",
        "correct": "Visit status.example.com for real-time uptime and incident history across all services.",
        "distractors": [
            "Internal dashboards show detailed metrics not exposed on the public status page.",
            "Scheduled maintenance windows are announced 48 hours in advance.",
            "You can subscribe to status updates via email or Slack webhook.",
        ],
    },
    {
        "category": "docs",
        "query": "what's the escalation policy if the on-call engineer doesn't respond?",
        "correct": "If the primary on-call doesn't acknowledge a page within 5 minutes, it escalates to the secondary on-call.",
        "distractors": [
            "The on-call rotation changes every week on Monday at 9am.",
            "Critical alerts page immediately; warnings are batched into a daily digest.",
            "On-call engineers get a stipend for weeks they're on rotation.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I report a bug?",
        "correct": "File a bug report in the issue tracker with steps to reproduce, expected behavior, and actual behavior.",
        "distractors": [
            "Feature requests go through a separate intake form for product review.",
            "Security vulnerabilities should be reported privately via the security email, not the public tracker.",
            "Bug reports are triaged within 2 business days.",
        ],
    },
    # -- billing --
    {
        "category": "docs",
        "query": "how do I upgrade my plan?",
        "correct": "Go to Billing Settings, select a new plan, and the change takes effect immediately with prorated charges.",
        "distractors": [
            "Downgrading takes effect at the start of the next billing cycle.",
            "Invoices are generated on the first of each month.",
            "You can add a purchase order number to invoices under Billing Settings.",
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
    {
        "category": "docs",
        "query": "how does usage-based billing work?",
        "correct": "You're billed monthly based on API calls made, with the first 10,000 calls free each month.",
        "distractors": [
            "Enterprise plans include a dedicated account manager.",
            "Unused seats on an annual plan don't roll over to the next year.",
            "You can view a real-time usage breakdown on the Billing dashboard.",
        ],
    },
    {
        "category": "docs",
        "query": "what payment methods are accepted?",
        "correct": "We accept credit cards and, for annual plans, wire transfer with a signed purchase order.",
        "distractors": [
            "Refunds for annual plans are prorated if canceled within 30 days.",
            "Failed payments retry automatically three times over a week.",
            "Tax is calculated automatically based on your billing address.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I cancel my subscription?",
        "correct": "Go to Billing Settings and click Cancel Subscription - your access continues until the end of the current billing period.",
        "distractors": [
            "Canceling doesn't delete your data; it's retained for 90 days.",
            "You can export your data at any time from Account Settings.",
            "Reactivating a canceled account restores your previous plan and data.",
        ],
    },
    # -- team / permissions --
    {
        "category": "docs",
        "query": "how do I set a team member's permission level?",
        "correct": "Go to Team Settings, click a member's name, and choose Admin, Editor, or Viewer from the role dropdown.",
        "distractors": [
            "Only Admins can invite new team members.",
            "Removing a member revokes their API keys immediately.",
            "Permission changes take effect the next time the member logs in.",
        ],
    },
    {
        "category": "docs",
        "query": "what can a Viewer role do?",
        "correct": "Viewers can see all data and reports but cannot create, edit, or delete anything.",
        "distractors": [
            "Editors can modify content but can't change team settings or billing.",
            "Admins have full access including billing and member management.",
            "Custom roles let you define granular permissions per resource type.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I transfer ownership of the account?",
        "correct": "Only the current owner can transfer ownership, from Account Settings, to another existing Admin.",
        "distractors": [
            "Deleting the account requires confirmation from the account owner.",
            "Owners are the only role that can close the account entirely.",
            "You can have only one owner per account at a time.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I remove a team member?",
        "correct": "Go to Team Settings, find the member, and click Remove - this immediately revokes their access and API keys.",
        "distractors": [
            "Removed members' created content is reassigned to the account owner.",
            "You can re-invite a removed member at any time.",
            "Removing the last Admin requires promoting another member first.",
        ],
    },
    {
        "category": "docs",
        "query": "can I restrict access by IP address?",
        "correct": "Enterprise plans support IP allowlisting under Security Settings to restrict login to specific ranges.",
        "distractors": [
            "Failed login attempts from blocked IPs are logged for review.",
            "VPN users may need to add their VPN's IP range to the allowlist.",
            "IP restrictions apply to the web dashboard but not the API.",
        ],
    },
    # -- data / backup --
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
        "query": "how do I restore from a backup?",
        "correct": "Contact support with the desired backup date - restores are performed manually and take up to 4 hours.",
        "distractors": [
            "You can download your own data export at any time without support.",
            "Backups are encrypted at rest with a key rotated quarterly.",
            "Point-in-time recovery is available on Enterprise plans, accurate to the minute.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I export my data?",
        "correct": "Go to Account Settings, click Export Data, and you'll receive a download link via email within an hour.",
        "distractors": [
            "Exports include all your data in CSV and JSON formats.",
            "Scheduled exports can be configured to run weekly and sent to an S3 bucket.",
            "Deleted items aren't included in a data export.",
        ],
    },
    {
        "category": "docs",
        "query": "what happens to my data if I delete my account?",
        "correct": "Account data is permanently deleted 30 days after account closure, giving you a window to reactivate.",
        "distractors": [
            "You can request immediate permanent deletion instead of the 30-day window.",
            "Some data may be retained longer if required for legal compliance.",
            "Backups containing your data are purged on the same retention schedule.",
        ],
    },
    {
        "category": "docs",
        "query": "how is data replicated across regions?",
        "correct": "Data is synchronously replicated to a secondary region within the same continent for durability.",
        "distractors": [
            "Cross-region replication for disaster recovery is available on Enterprise plans.",
            "You can choose your primary data region when creating an account.",
            "Replication lag is monitored and alerts fire if it exceeds 60 seconds.",
        ],
    },
    # -- monitoring --
    {
        "category": "docs",
        "query": "how do I set up a custom alert?",
        "correct": "Go to Monitoring, click New Alert, and define a metric threshold plus who gets notified.",
        "distractors": [
            "Default alerts are pre-configured for common failure scenarios.",
            "Alert history is retained for 90 days for review.",
            "You can silence an alert temporarily during planned maintenance.",
        ],
    },
    {
        "category": "docs",
        "query": "how often are metrics collected?",
        "correct": "System metrics are scraped every 15 seconds and aggregated into 1-minute buckets for dashboards.",
        "distractors": [
            "Custom application metrics can be sent via the metrics API.",
            "Dashboards support a maximum lookback window of 13 months.",
            "Metric data older than 13 months is downsampled to hourly resolution.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I view logs for a specific request?",
        "correct": "Search logs by the request ID shown in the X-Request-ID response header to trace a single request.",
        "distractors": [
            "Logs are searchable by timestamp, service name, and log level.",
            "Log retention is 30 days on standard plans, 1 year on Enterprise.",
            "You can export logs to your own SIEM via a streaming integration.",
        ],
    },
    {
        "category": "docs",
        "query": "what's included in the uptime SLA?",
        "correct": "The SLA guarantees 99.9 percent uptime monthly, with service credits issued if it's missed.",
        "distractors": [
            "Scheduled maintenance windows don't count against the uptime SLA.",
            "SLA credits are applied automatically to the next invoice.",
            "Enterprise plans can negotiate a custom SLA with higher guarantees.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I check API latency trends?",
        "correct": "The status page shows a rolling 24-hour p50/p95 latency chart for each API endpoint.",
        "distractors": [
            "Individual account latency isn't shown publicly for privacy reasons.",
            "You can query historical latency via the metrics API for your own account.",
            "Latency spikes trigger an internal investigation if sustained over 5 minutes.",
        ],
    },
    # -- integrations --
    {
        "category": "docs",
        "query": "how do I connect a Slack integration?",
        "correct": "Go to Integrations, click Slack, and authorize the app in your workspace to receive notifications.",
        "distractors": [
            "You can choose which event types get sent to Slack under integration settings.",
            "Removing the Slack integration doesn't delete past messages it sent.",
            "Multiple Slack workspaces can be connected on Enterprise plans.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I set up a webhook?",
        "correct": "Go to Integrations, click Add Webhook, enter your endpoint URL, and select which events trigger it.",
        "distractors": [
            "Webhook payloads are signed so you can verify they came from us.",
            "Failed webhook deliveries are retried up to 5 times with backoff.",
            "You can view webhook delivery logs for the past 7 days.",
        ],
    },
    {
        "category": "docs",
        "query": "does the product integrate with Zapier?",
        "correct": "Yes, our Zapier integration lets you trigger workflows from account events without writing code.",
        "distractors": [
            "We publish an official API client library for several languages.",
            "A native integration with Salesforce is available on Enterprise plans.",
            "Custom integrations can be built using our public API and webhooks.",
        ],
    },
    {
        "category": "docs",
        "query": "how do I import data from a spreadsheet?",
        "correct": "Go to Import, upload a CSV matching our template, and map columns to fields before confirming.",
        "distractors": [
            "Imports over 10,000 rows are processed asynchronously with an email when done.",
            "You can preview the first 10 rows before confirming an import.",
            "Failed rows during an import are listed in a downloadable error report.",
        ],
    },
    {
        "category": "docs",
        "query": "can I sync data with an external database?",
        "correct": "Enterprise plans support a one-way sync to an external Postgres database, updated hourly.",
        "distractors": [
            "Real-time sync via webhooks is available on all plans.",
            "You can request a one-time data export instead of ongoing sync.",
            "Sync credentials are managed separately from your main account login.",
        ],
    },
    # -- rate limits / quotas --
    {
        "category": "docs",
        "query": "what happens if I exceed my rate limit?",
        "correct": "Requests over the limit receive a 429 status code with a Retry-After header indicating when to try again.",
        "distractors": [
            "Rate limits reset at the top of every minute.",
            "Enterprise plans have configurable, higher rate limits.",
            "You can request a temporary rate limit increase for a planned traffic spike.",
        ],
    },
    {
        "category": "docs",
        "query": "how many API keys can I create?",
        "correct": "Each account can have up to 10 active API keys at a time.",
        "distractors": [
            "API keys don't expire unless manually revoked.",
            "You can name each API key to track what it's used for.",
            "Revoked keys stop working immediately, with no grace period.",
        ],
    },
    {
        "category": "docs",
        "query": "what's the storage quota on my plan?",
        "correct": "The standard plan includes 50GB of storage, with overage billed per additional GB.",
        "distractors": [
            "You'll get a warning email when you reach 80 percent of your quota.",
            "Deleted files count against your quota for 7 days before being purged.",
            "Enterprise plans have no fixed storage quota.",
        ],
    },
    {
        "category": "docs",
        "query": "how many team members can I add?",
        "correct": "The standard plan supports up to 25 team members; Enterprise plans are unlimited.",
        "distractors": [
            "Guest accounts with limited access don't count against your seat limit.",
            "You're billed per active seat, prorated for mid-cycle additions.",
            "Inactive members can be deactivated without losing their seat.",
        ],
    },
    {
        "category": "docs",
        "query": "is there a limit on how many projects I can create?",
        "correct": "Standard plans allow up to 20 projects; contact sales for higher limits.",
        "distractors": [
            "Archived projects don't count against your project limit.",
            "You can transfer a project between accounts if needed.",
            "Each project has its own separate API keys and settings.",
        ],
    },
    # ============================== CODE ==============================
    # -- validation --
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
        "query": "how is a phone number validated?",
        "correct": "def validate_phone(number: str, region: str = 'US') -> bool: uses libphonenumber to check if the number is valid for the given region.",
        "distractors": [
            "def format_phone(number: str) -> str: formats a raw phone number into a display-friendly string.",
            "def send_sms(number: str, message: str) -> None: sends an SMS via the configured provider.",
            "def normalize_phone(number: str) -> str: strips formatting characters from a phone number.",
        ],
    },
    {
        "category": "code",
        "query": "which function checks password strength?",
        "correct": "def check_password_strength(password: str) -> int: returns a score from 0-4 based on length, character variety, and common-password checks.",
        "distractors": [
            "def hash_password(password: str) -> str: hashes a password using bcrypt before storage.",
            "def generate_password(length: int = 16) -> str: generates a random secure password.",
            "def verify_password(password: str, hashed: str) -> bool: checks a plaintext password against a stored hash.",
        ],
    },
    {
        "category": "code",
        "query": "how are form inputs sanitized?",
        "correct": "def sanitize_input(text: str) -> str: strips HTML tags and control characters from user-submitted text before storage.",
        "distractors": [
            "def validate_required_fields(data: dict, fields: list) -> list: returns a list of missing required fields.",
            "def escape_sql(value: str) -> str: escapes special characters for raw SQL queries (prefer parameterized queries instead).",
            "def truncate_text(text: str, max_length: int) -> str: truncates text to a maximum length with an ellipsis.",
        ],
    },
    {
        "category": "code",
        "query": "which function validates a JSON schema?",
        "correct": "def validate_schema(data: dict, schema: dict) -> list: validates data against a JSON schema and returns a list of validation errors.",
        "distractors": [
            "def parse_json(raw: str) -> dict: parses a JSON string, raising a clear error on malformed input.",
            "def merge_dicts(a: dict, b: dict) -> dict: deep-merges two dictionaries, with b taking precedence.",
            "def flatten_dict(data: dict) -> dict: flattens a nested dictionary into dot-notation keys.",
        ],
    },
    # -- auth / security --
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
        "query": "which function checks if a user has permission?",
        "correct": "def has_permission(user, resource, action: str) -> bool: checks the user's role against the resource's access control list.",
        "distractors": [
            "def assign_role(user, role: str) -> None: assigns a role to a user.",
            "def list_permissions(user) -> list: returns all permissions granted to a user.",
            "def audit_log(actor, action: str, resource) -> None: records an action in the audit log.",
        ],
    },
    {
        "category": "code",
        "query": "how are API keys generated?",
        "correct": "def generate_api_key() -> str: generates a cryptographically random API key prefixed with the environment name.",
        "distractors": [
            "def revoke_api_key(key_id: str) -> None: marks an API key as revoked in the database.",
            "def list_api_keys(user) -> list: returns all active API keys for a user, with secrets masked.",
            "def rotate_api_key(key_id: str) -> str: generates a new key and schedules the old one for revocation.",
        ],
    },
    {
        "category": "code",
        "query": "which function rate-limits login attempts?",
        "correct": "def check_login_rate_limit(username: str) -> bool: returns False if the account has had 5+ failed attempts in the last 15 minutes.",
        "distractors": [
            "def record_login_attempt(username: str, success: bool) -> None: logs a login attempt for auditing.",
            "def unlock_account(username: str) -> None: manually clears a rate-limit lockout for an account.",
            "def send_login_alert(user, ip_address: str) -> None: emails the user about a login from a new device.",
        ],
    },
    {
        "category": "code",
        "query": "how is CSRF protection implemented?",
        "correct": "def verify_csrf_token(request) -> bool: compares the token in the request header against the one stored in the user's session.",
        "distractors": [
            "def generate_csrf_token(session) -> str: generates a new CSRF token and stores it in the session.",
            "def verify_origin(request) -> bool: checks the request's Origin header against an allowlist.",
            "def set_secure_cookie(response, name: str, value: str) -> None: sets a cookie with Secure and HttpOnly flags.",
        ],
    },
    # -- data access --
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
        "query": "how does the ORM query for active users?",
        "correct": "def get_active_users() -> list[User]: queries the users table for rows where status='active' and deleted_at IS NULL.",
        "distractors": [
            "def get_user_by_id(user_id: int) -> User | None: fetches a single user by primary key.",
            "def create_user(email: str, name: str) -> User: inserts a new user row and returns the created object.",
            "def soft_delete_user(user_id: int) -> None: sets deleted_at instead of removing the row.",
        ],
    },
    {
        "category": "code",
        "query": "which function handles database transactions?",
        "correct": "def transaction(): a context manager that commits on success and rolls back automatically if an exception is raised inside it.",
        "distractors": [
            "def execute_query(sql: str, params: tuple) -> list: runs a raw SQL query and returns the results.",
            "def bulk_insert(table: str, rows: list[dict]) -> None: inserts many rows in a single statement for efficiency.",
            "def get_row_count(table: str) -> int: returns the number of rows in a table.",
        ],
    },
    {
        "category": "code",
        "query": "how is a database connection pool configured?",
        "correct": "def create_pool(min_size=5, max_size=20) -> Pool: creates a connection pool with the given size bounds and a 30-second idle timeout.",
        "distractors": [
            "def close_pool(pool: Pool) -> None: gracefully closes all connections in a pool.",
            "def pool_stats(pool: Pool) -> dict: returns current pool utilization metrics.",
            "def health_check_pool(pool: Pool) -> bool: verifies the pool can acquire and release a connection.",
        ],
    },
    {
        "category": "code",
        "query": "which function reads from the cache before the database?",
        "correct": "def get_or_fetch(key: str, fetch_fn) -> Any: checks the cache first, falling back to fetch_fn and populating the cache on a miss.",
        "distractors": [
            "def invalidate_cache(key: str) -> None: removes a single key from the cache.",
            "def warm_cache(keys: list[str]) -> None: pre-populates the cache for a list of keys.",
            "def cache_stats() -> dict: returns hit/miss ratio and current cache size.",
        ],
    },
    # -- utility / formatting --
    {
        "category": "code",
        "query": "which function formats a currency amount?",
        "correct": "def format_currency(amount: float, currency: str = 'USD') -> str: formats a float as a currency string with the appropriate symbol and two decimal places.",
        "distractors": [
            "def parse_currency(text: str) -> float: parses a currency string back into a float.",
            "def convert_currency(amount: float, from_cur: str, to_cur: str) -> float: converts an amount using current exchange rates.",
            "def round_to_cents(amount: float) -> float: rounds a float to two decimal places.",
        ],
    },
    {
        "category": "code",
        "query": "how are dates formatted for display?",
        "correct": "def format_date(dt: datetime, style: str = 'short') -> str: formats a datetime according to the user's locale and preferred style.",
        "distractors": [
            "def parse_date(text: str) -> datetime: parses a date string in several common formats.",
            "def humanize_timedelta(delta: timedelta) -> str: converts a duration into a human-readable string like '3 days ago'.",
            "def convert_timezone(dt: datetime, tz: str) -> datetime: converts a datetime to a different timezone.",
        ],
    },
    {
        "category": "code",
        "query": "which function slugifies a string?",
        "correct": "def slugify(text: str) -> str: converts a string to a URL-safe slug by lowercasing, removing punctuation, and replacing spaces with hyphens.",
        "distractors": [
            "def truncate_text(text: str, max_length: int) -> str: truncates text to a maximum length with an ellipsis.",
            "def strip_html(text: str) -> str: removes HTML tags from a string, leaving plain text.",
            "def capitalize_words(text: str) -> str: capitalizes the first letter of each word in a string.",
        ],
    },
    {
        "category": "code",
        "query": "how is file size formatted for humans?",
        "correct": "def format_file_size(bytes: int) -> str: converts a byte count into a human-readable string like '4.2 MB'.",
        "distractors": [
            "def get_file_extension(filename: str) -> str: extracts the extension from a filename.",
            "def sanitize_filename(filename: str) -> str: removes characters that aren't safe for filenames.",
            "def compress_file(path: str) -> str: gzip-compresses a file and returns the new path.",
        ],
    },
    {
        "category": "code",
        "query": "which function generates a random ID?",
        "correct": "def generate_id(prefix: str = '') -> str: generates a URL-safe random ID, optionally prefixed, using a cryptographically secure generator.",
        "distractors": [
            "def generate_slug(text: str) -> str: generates a URL-safe slug from a title.",
            "def hash_content(content: str) -> str: computes a SHA-256 hash of the given content.",
            "def generate_uuid() -> str: generates a standard UUID4 string.",
        ],
    },
    # -- async / concurrency --
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
        "query": "which function runs tasks concurrently?",
        "correct": "async def run_concurrent(tasks: list, max_parallel=10): runs a list of async tasks with a semaphore capping concurrency at max_parallel.",
        "distractors": [
            "async def run_sequential(tasks: list): runs a list of async tasks one after another.",
            "def run_in_thread(fn, *args): runs a blocking function in a background thread pool.",
            "async def gather_with_timeout(tasks: list, timeout: float): runs tasks concurrently, canceling any that exceed the timeout.",
        ],
    },
    {
        "category": "code",
        "query": "how are background jobs queued?",
        "correct": "def enqueue_job(job_type: str, payload: dict, delay: int = 0) -> str: adds a job to the queue, optionally delayed, and returns the job ID.",
        "distractors": [
            "def get_job_status(job_id: str) -> str: returns the current status of a queued or running job.",
            "def cancel_job(job_id: str) -> bool: cancels a queued job if it hasn't started yet.",
            "def process_job_queue(): the worker loop that pulls jobs off the queue and executes them.",
        ],
    },
    {
        "category": "code",
        "query": "which function debounces rapid calls?",
        "correct": "def debounce(fn, wait_ms: int): wraps a function so it only actually runs after wait_ms has passed with no new calls.",
        "distractors": [
            "def throttle(fn, calls_per_second: float): wraps a function to limit how often it can be called.",
            "def memoize(fn): caches a function's return value based on its arguments.",
            "def batch_calls(fn, batch_size: int): groups individual calls into batches before executing.",
        ],
    },
    {
        "category": "code",
        "query": "how is a distributed lock acquired?",
        "correct": "def acquire_lock(key: str, timeout: float = 10) -> Lock: acquires a distributed lock via Redis, blocking up to timeout seconds.",
        "distractors": [
            "def release_lock(lock: Lock) -> None: releases a previously acquired distributed lock.",
            "def is_locked(key: str) -> bool: checks whether a given lock key is currently held.",
            "def extend_lock(lock: Lock, extra_seconds: float) -> None: extends the expiry of an already-held lock.",
        ],
    },
    # -- error handling --
    {
        "category": "code",
        "query": "which function handles uncaught exceptions?",
        "correct": "def global_exception_handler(exc_type, exc_value, traceback): logs uncaught exceptions with full context before the process exits.",
        "distractors": [
            "def capture_exception(exc: Exception, context: dict = None): manually reports an exception to the error tracking service.",
            "def suppress_errors(fn): wraps a function to catch and log exceptions without re-raising.",
            "def format_traceback(exc: Exception) -> str: formats an exception's traceback as a readable string.",
        ],
    },
    {
        "category": "code",
        "query": "how are validation errors collected?",
        "correct": "class ValidationErrors: accumulates multiple field-level errors during form validation so they can all be returned at once.",
        "distractors": [
            "class APIError(Exception): the base exception class for all API-level errors.",
            "class RetryableError(Exception): marks an error as safe to retry automatically.",
            "class ConfigError(Exception): raised when required configuration is missing or invalid.",
        ],
    },
    {
        "category": "code",
        "query": "which function implements a circuit breaker?",
        "correct": "class CircuitBreaker: trips open after N consecutive failures, rejecting calls immediately until a cooldown period passes.",
        "distractors": [
            "class RateLimiter: enforces a maximum number of calls per time window.",
            "class RetryPolicy: defines how many times and how long to wait between retries.",
            "class Timeout: wraps a call with a maximum execution time before raising.",
        ],
    },
    {
        "category": "code",
        "query": "how does the fallback logic work when a service is down?",
        "correct": "def with_fallback(primary_fn, fallback_fn): calls primary_fn, falling back to fallback_fn if it raises or times out.",
        "distractors": [
            "def health_check(service: str) -> bool: pings a service to check if it's currently healthy.",
            "def mark_unhealthy(service: str) -> None: flags a service as unhealthy to stop routing traffic to it.",
            "def degrade_gracefully(feature: str) -> bool: checks whether a non-critical feature should be disabled under load.",
        ],
    },
    {
        "category": "code",
        "query": "which function validates configuration on startup?",
        "correct": "def validate_config(config: dict) -> None: raises a ConfigError listing every missing or invalid setting before the app starts.",
        "distractors": [
            "def load_config(path: str) -> dict: loads configuration from a file, merging in environment variable overrides.",
            "def get_config_value(key: str, default=None): fetches a single config value with an optional default.",
            "def reload_config() -> None: hot-reloads configuration without restarting the process.",
        ],
    },
    # -- search / ranking --
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
    {
        "category": "code",
        "query": "which function handles fuzzy string matching?",
        "correct": "def fuzzy_match(query: str, candidates: list[str]) -> list[tuple[str, float]]: returns candidates with a similarity score using edit distance.",
        "distractors": [
            "def exact_match(query: str, candidates: list[str]) -> list[str]: returns candidates that exactly equal the query.",
            "def tokenize(text: str) -> list[str]: splits text into search tokens, lowercased and stripped of punctuation.",
            "def stem_word(word: str) -> str: reduces a word to its root form for matching.",
        ],
    },
    {
        "category": "code",
        "query": "how is search relevance combined from multiple signals?",
        "correct": "def fuse_scores(bm25_scores, vector_scores, weights=(0.5, 0.5)): combines two ranked lists into one score using reciprocal rank fusion.",
        "distractors": [
            "def normalize_scores(scores: list[float]) -> list[float]: scales a list of scores into the 0-1 range.",
            "def boost_recent(results, half_life_days: float): boosts the score of more recently created results.",
            "def filter_by_threshold(results, min_score: float): drops results below a minimum relevance score.",
        ],
    },
    {
        "category": "code",
        "query": "which function autocompletes search queries?",
        "correct": "def autocomplete(prefix: str, max_results: int = 5) -> list[str]: returns the most popular past queries starting with the given prefix.",
        "distractors": [
            "def spell_correct(query: str) -> str: suggests a corrected spelling for a likely-misspelled query.",
            "def expand_query(query: str) -> list[str]: generates related query variants to broaden a search.",
            "def log_search_query(query: str, user_id: str) -> None: records a search query for analytics.",
        ],
    },
    {
        "category": "code",
        "query": "how are duplicate search results removed?",
        "correct": "def dedupe_by_similarity(results, threshold: float = 0.9): drops results whose embeddings are near-identical to a higher-ranked result.",
        "distractors": [
            "def dedupe_by_id(results): removes results with a duplicate unique ID, keeping the first occurrence.",
            "def merge_duplicate_sources(results): combines results pointing to the same underlying document from different indexes.",
            "def limit_per_source(results, max_per_source: int): caps how many results can come from a single source.",
        ],
    },
    # -- caching --
    {
        "category": "code",
        "query": "which function implements the caching layer?",
        "correct": "def cache_result(fn): memoizes a function's return value based on its arguments, with a configurable TTL.",
        "distractors": [
            "def invalidate_cache(pattern: str) -> int: removes all cache keys matching a pattern, returning the count removed.",
            "def cache_stats() -> dict: returns hit rate, miss rate, and current cache size.",
            "def warm_cache(keys: list[str]) -> None: pre-populates cache entries before traffic arrives.",
        ],
    },
    {
        "category": "code",
        "query": "how does cache invalidation work on write?",
        "correct": "def invalidate_on_write(model_instance): clears any cache keys derived from a model after it's saved or deleted.",
        "distractors": [
            "def set_cache_ttl(key: str, seconds: int) -> None: overrides the default TTL for a specific cache key.",
            "def get_cache_key(model, id) -> str: builds the canonical cache key for a given model instance.",
            "def bypass_cache(fn): decorates a function to skip the cache for this call only.",
        ],
    },
    {
        "category": "code",
        "query": "which function evicts stale cache entries?",
        "correct": "def evict_expired(): a background task that scans the cache and removes entries past their TTL.",
        "distractors": [
            "def evict_lru(max_size: int): removes the least-recently-used entries when the cache exceeds max_size.",
            "def preload_cache(keys: list[str]): warms the cache for a known set of keys at startup.",
            "def clear_all_cache(): flushes the entire cache, used mainly during deploys.",
        ],
    },
    {
        "category": "code",
        "query": "how is the cache backend configured?",
        "correct": "def get_cache_backend() -> CacheBackend: returns a Redis-backed cache client, falling back to an in-memory dict in tests.",
        "distractors": [
            "def set_cache_backend(backend: CacheBackend) -> None: overrides the cache backend, mainly for testing.",
            "def cache_backend_health() -> bool: checks whether the configured cache backend is reachable.",
            "def migrate_cache_backend(old, new) -> None: copies entries from one cache backend to another.",
        ],
    },
    {
        "category": "code",
        "query": "which function caches API responses?",
        "correct": "def cached_response(ttl: int = 60): a decorator for route handlers that caches the response body for ttl seconds.",
        "distractors": [
            "def cache_control_header(ttl: int) -> str: builds a Cache-Control header value for HTTP responses.",
            "def etag_for(content: bytes) -> str: computes an ETag hash for HTTP conditional requests.",
            "def vary_by_header(header_name: str): decorates a route so its cache varies by a specific request header.",
        ],
    },
    # -- testing --
    {
        "category": "code",
        "query": "which function creates test fixtures?",
        "correct": "def make_test_user(**overrides) -> User: creates a User instance with sensible defaults, overridable per test.",
        "distractors": [
            "def cleanup_test_data(): deletes all data created during a test run.",
            "def seed_test_database(): populates the test database with a standard baseline dataset.",
            "def mock_external_api(): patches outgoing HTTP calls to return canned responses during tests.",
        ],
    },
    {
        "category": "code",
        "query": "how are database tests isolated from each other?",
        "correct": "def transactional_test_case(): a base test class that wraps each test in a transaction rolled back after it finishes.",
        "distractors": [
            "def reset_database(): truncates all tables and re-seeds baseline data between test suites.",
            "def use_test_database(): points the app at a separate database instance for the test run.",
            "def snapshot_database(): captures the current database state to restore later.",
        ],
    },
    {
        "category": "code",
        "query": "which function mocks the current time in tests?",
        "correct": "def freeze_time(dt: datetime): a context manager that makes datetime.now() return a fixed value during the test.",
        "distractors": [
            "def travel_forward(days: int): advances the frozen test clock by a number of days.",
            "def mock_random_seed(seed: int): fixes the random number generator's seed for deterministic tests.",
            "def fast_forward_jobs(): immediately runs any jobs scheduled for the future during a test.",
        ],
    },
    {
        "category": "code",
        "query": "how do integration tests hit a real HTTP server?",
        "correct": "def test_client() -> TestClient: spins up the app in-process and returns a client for making real HTTP requests against it.",
        "distractors": [
            "def mock_http_client(): returns a client that returns canned responses instead of making real requests.",
            "def record_http_interactions(): records real HTTP responses to replay in future test runs.",
            "def assert_response_matches(response, expected): asserts a response body matches an expected structure.",
        ],
    },
    {
        "category": "code",
        "query": "which function asserts an exception was raised with a specific message?",
        "correct": "def assert_raises_with_message(exc_type, message: str, fn, *args): calls fn and asserts it raises exc_type with the given message.",
        "distractors": [
            "def assert_called_with(mock, *args, **kwargs): asserts a mock was called with specific arguments.",
            "def assert_eventually(condition_fn, timeout: float): polls a condition until it's true or a timeout is reached.",
            "def capture_logs(): a context manager that captures log output for assertions in a test.",
        ],
    },
    # -- API clients --
    {
        "category": "code",
        "query": "which function wraps outgoing HTTP requests?",
        "correct": "def api_request(method: str, path: str, **kwargs) -> Response: wraps the HTTP client with automatic retries, auth headers, and error handling.",
        "distractors": [
            "def build_url(base: str, path: str, params: dict) -> str: constructs a full URL from a base, path, and query parameters.",
            "def parse_response(response) -> dict: parses and validates a JSON API response, raising on error status codes.",
            "def get_default_headers() -> dict: returns the default headers sent with every outgoing request.",
        ],
    },
    {
        "category": "code",
        "query": "how does the client handle pagination automatically?",
        "correct": "def iter_all_pages(endpoint: str): a generator that yields items across every page of a paginated API endpoint automatically.",
        "distractors": [
            "def get_page(endpoint: str, cursor: str = None): fetches a single page of results from a paginated endpoint.",
            "def count_total_items(endpoint: str) -> int: fetches the total item count without retrieving all pages.",
            "def get_page_size(endpoint: str) -> int: returns the configured page size for a given endpoint.",
        ],
    },
    {
        "category": "code",
        "query": "which function handles API client authentication?",
        "correct": "def authenticate_client(api_key: str) -> Client: constructs an API client pre-configured with the given key in its default headers.",
        "distractors": [
            "def refresh_client_token(client: Client) -> None: refreshes an expiring OAuth token used by the client.",
            "def validate_client_credentials(api_key: str) -> bool: checks whether a given API key is valid before use.",
            "def revoke_client_session(client: Client) -> None: invalidates the client's current session on the server.",
        ],
    },
    {
        "category": "code",
        "query": "how are webhook payloads verified as authentic?",
        "correct": "def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool: recomputes the HMAC signature and compares it to the one provided.",
        "distractors": [
            "def parse_webhook_event(payload: bytes) -> dict: parses a webhook payload into a structured event object.",
            "def dispatch_webhook_event(event: dict) -> None: routes a parsed webhook event to the appropriate handler.",
            "def retry_failed_webhook(event_id: str) -> None: manually retries delivery of a previously failed webhook.",
        ],
    },
    {
        "category": "code",
        "query": "which function handles API client timeouts?",
        "correct": "def request_with_timeout(method: str, url: str, timeout: float = 5.0): makes an HTTP request that raises a TimeoutError if it exceeds the given duration.",
        "distractors": [
            "def request_with_retry(method: str, url: str, max_attempts: int = 3): retries a failed request with backoff.",
            "def request_with_circuit_breaker(method: str, url: str): wraps a request with circuit-breaker protection against a failing endpoint.",
            "def configure_connection_pool(max_connections: int): sets the maximum number of concurrent connections the client can hold open.",
        ],
    },
]
