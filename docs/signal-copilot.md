# SIGNAL - AI product strategist

**Design:** deterministic intent router → analytical tools over the visitor's own session state → structured
`Recommendation` (recommendation, evidence, rationale, confidence, trade-offs, counterargument, experiment, success
metric, owner, next action) + cited record IDs + calculation references. Personality lives only in the one-line headline,
chosen from the evidence (e.g. weak evidence → "That's a hypothesis, not a roadmap").

| Question | Tool | Real function invoked |
|---|---|---|
| What should we build next? | build_next | ranked opportunities, `explain_outranks` |
| Which three issues threaten the most revenue? | revenue_threats | unique-account exposure |
| Why is this cluster ranked first? | why_ranked | priority arithmetic, `tornado` |
| What happens if engineering capacity drops 30%? | capacity_change | `optimise` twice (MILP) |
| What would a CFO / CTO / VP Product / CS / skeptic prioritise? | persona | lens over the same evidence |
| Run the decision debate | debate | five lenses + reconciled vote |
| Which recommendations have weak evidence? | weak_evidence | evidence sufficiency rule |
| What are we ignoring? | ignoring | severity + volume-normalised trend |
| Which feature should we kill? | kill | argmin priority |
| Highest ROI? | highest_roi | ROI |
| Challenge this roadmap / What would reverse this? | challenge / reversal | alternative plan, assumption flips, Monte Carlo, tornado |
| Write a PRD / memo | prd / memo | routes to generated documents with the same numbers |

**LLM (optional):** set `ANTHROPIC_API_KEY` as a server secret to have answers narrated; the model only receives the
tool output as JSON and is instructed to add no numbers. Without a key everything works deterministically.
**Guardrails:** customer text is quoted data; instruction-like phrases are redacted and flagged; tests assert injected
feedback cannot change rankings and that keys never render.
