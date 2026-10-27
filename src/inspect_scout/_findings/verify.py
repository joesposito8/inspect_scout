"""Verify model quotations against original messages, then relocate miscitations."""

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel

from inspect_scout._llm_scanner.rendering import ScannerRendering
from inspect_scout._scanner.extract import message_as_str, message_numbering
from inspect_scout._scanner.result import Reference
from inspect_scout._transcript.types import Transcript


class VerifiedQuote(BaseModel):
    """The claimed and located positions of one quotation."""

    model_ref: str | None
    verified_ref: str | None
    status: Literal["verified", "relocated", "unlocated"]
    message_index: int | None = None
    message_label: str | None = None


def excerpt(text: str, nonce: str | None = None) -> str:
    """Keep a short quotation without rendering nonce markers."""
    if nonce:
        text = text.replace(f" ({nonce})", "").replace(f" {nonce}>", ">")
    return text[:200]


# Models re-type quotation marks and dashes, and drop or add markdown emphasis,
# when copying a passage; neither changes which words were quoted.
_EQUIVALENT_CHARACTERS = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": "'",
        "”": "'",
        '"': "'",
        "–": "-",
        "—": "-",
        "−": "-",
    }
)
_MARKDOWN = re.compile(r"[*_`~#]")


def normalize_quote_text(text: str) -> str:
    """Normalize text so that quotations match regardless of typography."""
    text = unicodedata.normalize("NFKC", text).translate(_EQUIVALENT_CHARACTERS)
    return " ".join(_MARKDOWN.sub("", text).split()).casefold()


def quote_fragments(quote: str) -> list[str]:
    """Split a quotation at ellipses into normalized fragments."""
    fragments = (
        normalize_quote_text(fragment).strip(" .,;:!?")
        for fragment in quote.replace("…", "...").split("...")
    )
    return [fragment for fragment in fragments if fragment]


def fragments_match(fragments: list[str], normalized_text: str) -> bool:
    """Find the fragments in order within text already passed through normalize_quote_text."""
    if not fragments:
        return False
    offset = 0
    for fragment in fragments:
        position = normalized_text.find(fragment, offset)
        if position == -1:
            return False
        offset = position + len(fragment)
    return True


def quote_matches(quote: str, text: str) -> bool:
    """Match ordered quote fragments, ignoring typography, whitespace and letter case."""
    return fragments_match(quote_fragments(quote), normalize_quote_text(text))


class QuoteVerifier:
    """Rendered messages and both numberings for repeated checks on a transcript."""

    def __init__(self, transcript: Transcript, rendering: ScannerRendering) -> None:
        self.transcript = transcript
        self.rendering = rendering
        self.indices = {
            message.id: index
            for index, message in enumerate(transcript.messages, 1)
            if message.id is not None
        }
        self.texts = {
            message.id: text
            for message in transcript.messages
            if message.id is not None
            and (
                text := message_as_str(
                    message, rendering.preprocessor(), nonce=rendering.nonce
                )
            )
            is not None
        }
        self.labels: dict[str, str | None] = {}
        self.normalized: dict[str, str] = {}

    def excerpt(self, message_id: str) -> str:
        """Render a bounded fallback for a legacy or not-yet-judged citation."""
        message = self.transcript.messages[self.indices[message_id] - 1]
        return excerpt(message_as_str(message, self.rendering.preprocessor()) or "")

    async def initialize(self) -> None:
        """Build Scout's numbering once using the scan's saved rendering settings."""
        render, _, label = message_numbering(
            self.rendering.preprocessor(), label_for_id=True, nonce=self.rendering.nonce
        )
        await render(self.transcript.messages)
        self.labels = {message_id: label(message_id) for message_id in self.texts}
        # Only numbered messages survived the exact rendering's role transform.
        self.texts = {
            message_id: text
            for message_id, text in self.texts.items()
            if self.labels[message_id] is not None
        }
        self.normalized = {
            message_id: normalize_quote_text(text)
            for message_id, text in self.texts.items()
        }

    def verify(self, quote: str, references: list[Reference]) -> VerifiedQuote:
        """Verify a cited message, or find the closest matching message."""
        fragments = quote_fragments(quote)
        refs = [ref for ref in references if ref.type == "message"]
        for ref in refs:
            if ref.id in self.normalized and fragments_match(
                fragments, self.normalized[ref.id]
            ):
                return VerifiedQuote(
                    model_ref=ref.id,
                    verified_ref=ref.id,
                    status="verified",
                    message_index=self.indices[ref.id],
                    message_label=self.labels.get(ref.id),
                )
        model_ref = refs[0].id if refs else None
        matches = [
            message_id
            for message_id, text in self.normalized.items()
            if fragments_match(fragments, text)
        ]
        if not matches:
            return VerifiedQuote(
                model_ref=model_ref, verified_ref=None, status="unlocated"
            )
        anchor = self.indices.get(model_ref, 1) if model_ref is not None else 1
        verified = min(
            matches,
            key=lambda message_id: (
                abs(self.indices[message_id] - anchor),
                self.indices[message_id],
            ),
        )
        return VerifiedQuote(
            model_ref=model_ref,
            verified_ref=verified,
            status="relocated",
            message_index=self.indices[verified],
            message_label=self.labels.get(verified),
        )
