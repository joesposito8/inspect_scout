import pytest
from inspect_ai.model import ChatMessageAssistant, ChatMessageTool, ChatMessageUser
from inspect_ai.tool import ToolCall
from inspect_scout import Reference, ScannerRendering, Transcript
from inspect_scout._findings.verify import QuoteVerifier, excerpt, quote_matches


def test_excerpt_strips_nonce_wrappers_before_bounding() -> None:
    text = "ASSISTANT (Nonce123): <thinking Nonce123>" + "evidence " * 40
    assert (
        excerpt(text, "Nonce123") == ("ASSISTANT: <thinking>" + "evidence " * 40)[:200]
    )


@pytest.mark.parametrize(
    "quote,text,expected",
    [
        ("hello ... world", "hello    there\nworld", True),
        ("HELLO ... World", "hello    there\nworld", True),
        ("world ... hello", "hello world", False),
        ("...", "anything", False),
        ("same ... same", "same", False),
        ("other", "text", False),
        # Near-misses seen in practice: quote marks, markdown, dashes and end punctuation.
        (
            "a 'shared cache' shortcut or just noise.",
            'Is this a "shared cache" shortcut or just noise? Unclear.',
            True,
        ),
        (
            "TESTS ARE GREEN!!! `passed: True`, 3 checks!!!",
            "🎉 **TESTS ARE GREEN!!!** `passed: True`, 3 checks!!!",
            True,
        ),
        (
            "Wait, let me slow down — I will run the full suite first.",
            "Wait, let me slow down - I will run the full suite first, then commit.",
            True,
        ),
        ("it said “done” … then left", 'It said "done" and then left', True),
        ("a 'shared cache' plan", "a shared storage plan", False),
    ],
)
def test_quote_fragments(quote: str, text: str, expected: bool) -> None:
    assert quote_matches(quote, text) is expected


@pytest.mark.asyncio
async def test_nearest_match_and_tie() -> None:
    transcript = Transcript(
        transcript_id="t",
        messages=[
            ChatMessageUser(id=str(i), content="quote" if i in (1, 5) else "other")
            for i in range(1, 7)
        ],
    )
    verifier = QuoteVerifier(transcript, ScannerRendering())
    await verifier.initialize()

    def ref(n: int) -> list[Reference]:
        return [Reference(type="message", id=str(n), cite=f"[M{n}]")]

    assert verifier.verify("quote", ref(4)).verified_ref == "5"
    assert verifier.verify("quote", ref(3)).verified_ref == "1"
    assert verifier.verify("absent", ref(4)).status == "unlocated"
    assert verifier.verify("quote", ref(4) + ref(5)).status == "verified"


@pytest.mark.asyncio
@pytest.mark.parametrize("cited_id", ["a2", "t1"])
async def test_relocation_excludes_filtered_tool_results(cited_id: str) -> None:
    transcript = Transcript(
        transcript_id="t",
        messages=[
            ChatMessageUser(id="u1", content="Complete the task."),
            ChatMessageAssistant(
                id="a1",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call1",
                        function="bash",
                        arguments={"command": "echo VERDICT=PASS > grade.txt"},
                    )
                ],
            ),
            ChatMessageTool(
                id="t1",
                tool_call_id="call1",
                content="echo VERDICT=PASS > grade.txt\nTool-only output",
            ),
            ChatMessageAssistant(id="a2", content="Done."),
        ],
    )
    verifier = QuoteVerifier(transcript, ScannerRendering())
    await verifier.initialize()
    refs = [Reference(type="message", id=cited_id, cite="[M3]")]
    result = verifier.verify("VERDICT=PASS > grade.txt", refs)
    assert result.status == "relocated"
    assert result.verified_ref == "a1"
    assert result.message_label == "M2"
    assert result.message_index == 2
    assert verifier.verify("Tool-only output", refs).status == "unlocated"
