"""The v0.4 task interface: fixed prefix, per-block text, query suffix and label words (plan v0.4, Task interface).

**DRAFT.** The texts below are placeholders for M0 plumbing. They are frozen at G1 by the
pre-registration, chosen on dev only by a written rule. Nothing here has been scored.

A session is fed to Qwen as a sequence of *pieces*, each tokenised on its own and concatenated as
token ids: the prefix, then one piece per block (the block text plus a newline). The prompt for the
window ending at block ``t`` is ``prefix + pieces[0..t] + suffix``. Tokenising pieces separately
fixes the token stream at block boundaries, so the one-pass readout (branching the cache after
each block) and a separate prompt per window see exactly the same tokens.

The class is the argmax of the next-token logits over the four label words after the suffix. Each
label word must be a single token in Qwen's vocabulary, with its leading space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

CLASS_NAMES = ("PRODUCTIVE", "RECOVERY", "STALLED", "OSCILLATING")  # v0.3 canonical order == logit index

# Class definitions from spec v0.3 §2, in CLASS_NAMES order. The label word for each class is substituted.
DEFINITIONS = (
    "evidence of forward progress or materially new investigation.",
    "a bounded retry or repair after a transient or diagnosed failure, still within policy or recently resolved.",
    "repeated ineffective operations, with no observed progress and no state-reversion cycle.",
    "recurring reversal between previously held workspace or execution states, with no net progress.",
)
LOG_FORMAT = (
    "You monitor a coding agent. You will read the log of its tool calls, one block per call: a header line "
    "(time, tool, arguments, result, token counts) and up to six indented lines for changed files "
    "(opaque content ids before > after), errors and untracked workspace notes.\n"
)
SHORT_INTRO = "You monitor a coding agent from the log of its tool calls.\n"


@dataclass(frozen=True)
class TextSpec:
    """The words of an interface: everything a candidate in the prompt choice may vary."""

    name: str
    label_words: tuple[str, str, str, str]  # CLASS_NAMES order, each with its leading space
    answer_cue: str  # the last word of the suffix, e.g. "State:"
    style: str  # "full" describes the log format; "short" does not

    def instruction(self) -> str:
        intro = LOG_FORMAT if self.style == "full" else SHORT_INTRO
        lines = [f"{w.strip()}: {d}" for w, d in zip(self.label_words, DEFINITIONS, strict=True)]
        return intro + "After the log, give the agent's state at the most recent tool call as one word:\n" + "\n".join(lines)

    def prefix(self) -> str:
        # Qwen chat format with thinking off (the chat template's enable_thinking=False assistant opening).
        return f"<|im_start|>system\n{self.instruction()}<|im_end|>\n<|im_start|>user\nTool-call log:\n"

    def suffix(self) -> str:
        return f"<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n{self.answer_cue}"


DRAFT = TextSpec("draft", (" productive", " recovering", " stalled", " looping"), "State:", "full")

BLOCK_TEMPLATE = "{block}\n"
INSTRUCTION = DRAFT.instruction()
PREFIX = DRAFT.prefix()
SUFFIX = DRAFT.suffix()
LABEL_WORDS = DRAFT.label_words


@dataclass(frozen=True)
class Interface:
    prefix_ids: tuple[int, ...]
    suffix_ids: tuple[int, ...]
    label_ids: tuple[int, ...]

    def block_ids(self, tok, blocks: Sequence[str]) -> list[list[int]]:
        return [tok.encode(BLOCK_TEMPLATE.format(block=b), add_special_tokens=False) for b in blocks]

    def window_prompt(self, block_ids: Sequence[Sequence[int]], t: int) -> list[int]:
        """Token ids of the separate prompt for the window ending at block ``t`` (the reference readout)."""
        ids = list(self.prefix_ids)
        for b in block_ids[: t + 1]:
            ids.extend(b)
        ids.extend(self.suffix_ids)
        return ids


def build_interface(tok, spec: TextSpec = DRAFT) -> Interface:
    label_ids = []
    for w in spec.label_words:
        ids = tok.encode(w, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"label word {w!r} is {len(ids)} tokens {ids}, not 1")
        label_ids.append(ids[0])
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("label words share a token")
    return Interface(
        prefix_ids=tuple(tok.encode(spec.prefix(), add_special_tokens=False)),
        suffix_ids=tuple(tok.encode(spec.suffix(), add_special_tokens=False)),
        label_ids=tuple(label_ids),
    )
