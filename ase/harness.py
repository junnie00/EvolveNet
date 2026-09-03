"""The coder's HARNESS — the rewritable shell around the frozen model (Phase A: prompt-level).

The whole point of the project: the proposer agent IMPROVES the coder by rewriting this harness,
autonomously. It is NOT a fixed menu of toggles (that was the old AgentConfig/ladder). In Phase A
the harness is the prompt itself — a system prompt + a user-message template with {schema} and
{question} slots. The proposer can put ANY instructions / reasoning steps / schema-reading guidance
/ output rules into it. Phases B/C will let the proposer change real pipeline code (retrieval,
voting, repair), not just text.
"""
from dataclasses import dataclass, asdict


@dataclass
class Harness:
    system_prompt: str
    user_template: str          # MUST contain {schema} and {question}

    def render(self, schema, question):
        # token-replace (not str.format) so stray { } the model emits can't crash rendering
        return self.user_template.replace("{schema}", schema).replace("{question}", question)

    def to_dict(self):
        return asdict(self)


# the bare baseline coder = exactly the old default Solver behaviour, expressed as a harness
DEFAULT_HARNESS = Harness(
    system_prompt=("You are an expert at translating natural-language questions into SQLite SQL. "
                   "Return exactly one SQL query inside a ```sql ... ``` block, nothing else."),
    user_template="Database schema:\n{schema}\n\nQuestion: {question}",
)


def from_dict(d):
    return Harness(system_prompt=d["system_prompt"], user_template=d["user_template"])
