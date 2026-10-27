from typing import Any

import pytest
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    ModelOutput,
    get_model,
)
from inspect_ai.model._providers.anthropic import AnthropicAPI
from inspect_ai.tool import ToolCall, ToolChoice, ToolInfo
from inspect_scout import ScannerRendering, Transcript, finding_scanner, llm_scanner
from inspect_scout._scanner.result import Result


class Capture:
    def __init__(self, items: list[dict[str, Any]] | None = None) -> None:
        self.requests: list[
            tuple[list[ChatMessage], list[ToolInfo], GenerateConfig]
        ] = []
        self.items = items or []

    def __call__(
        self,
        messages: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        self.requests.append((list(messages), tools, config))
        if tools:
            return ModelOutput.from_message(
                ChatMessageAssistant(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="answer1",
                            function="answer",
                            arguments={"results": self.items},
                        )
                    ],
                )
            )
        return ModelOutput.from_content(model="mockllm", content="Reason.\nANSWER: yes")


@pytest.mark.asyncio
async def test_shared_prefix_and_tools() -> None:
    transcript = Transcript(
        transcript_id="shared",
        messages=[
            ChatMessageUser(id=f"m{i}", content="word " * 100) for i in range(300)
        ],
    )
    rendering = ScannerRendering(context_window=20000, nonce="TestNonce123")
    captures: list[Capture] = []
    for prompt, labels in [
        ("Is this useful?", ["yes"]),
        ("filler " * 300 + "Is this useful?", None),
    ]:
        capture = Capture()
        captures.append(capture)
        model = get_model("mockllm/model", custom_outputs=capture, memoize=False)
        await finding_scanner(prompt, labels, "test", rendering=rendering, model=model)(
            transcript
        )
    api = AnthropicAPI(model_name="claude-haiku-5-5", api_key="sk-test")
    requests = []
    for capture in captures:
        converted = []
        for messages, tools, config in capture.requests:
            system, tool_params, _, message_params, _ = await api.resolve_chat_input(
                input=messages, tools=tools, config=config, cache_ttl=None
            )
            content = message_params[0]["content"]
            assert isinstance(content, list)
            converted.append((system, tool_params, content[0]))
        requests.append(converted)
    assert len(requests[0]) > 1
    assert requests[0] == requests[1]

    purpose_capture = Capture()
    purpose_model = get_model(
        "mockllm/model", custom_outputs=purpose_capture, memoize=False
    )
    await llm_scanner(
        question="Is this useful?",
        answer="boolean",
        model=purpose_model,
        rendering=rendering,
    )(transcript)
    messages, tools, config = purpose_capture.requests[0]
    system, tool_params, _, message_params, _ = await api.resolve_chat_input(
        input=messages, tools=tools, config=config, cache_ttl=None
    )
    content = message_params[0]["content"]
    assert isinstance(content, list)
    assert content[0] == requests[0][0][2]
    assert tool_params != requests[0][0][1]


@pytest.mark.asyncio
async def test_roles_references_and_quote() -> None:
    capture = Capture(
        [
            {
                "quote": "forged PASS",
                "reason": "[M2] admits falsification",
                "label": "gamed",
            }
        ]
    )
    model = get_model("mockllm/model", custom_outputs=capture, memoize=False)
    transcript = Transcript(
        transcript_id="roles",
        messages=[
            ChatMessageUser(id="u", content="Task"),
            ChatMessageTool(id="t", content="tool secret", tool_call_id="c"),
            ChatMessageAssistant(
                id="a",
                content="I forged PASS",
                tool_calls=[
                    ToolCall(id="c", function="write", arguments={"grade": "PASS"})
                ],
            ),
        ],
    )
    result = await finding_scanner(
        "Find cheating",
        ["gamed"],
        "sha",
        rendering=ScannerRendering(nonce="TestNonce123"),
        model=model,
    )(transcript)
    assert isinstance(result, Result) and isinstance(result.value, list)
    item = Result.model_validate(result.value[0])
    assert item.references[0].id == "a"
    assert item.label == "gamed"
    text = str(capture.requests[0][0][0].content)
    assert "tool secret" not in text
    assert "Tool Call (TestNonce123): write" in text


@pytest.mark.asyncio
async def test_shared_rendering_overflow() -> None:
    capture = Capture()
    model = get_model("mockllm/model", custom_outputs=capture, memoize=False)
    judge = llm_scanner(
        question="filler " * 3000,
        answer="boolean",
        model=model,
        rendering=ScannerRendering(context_window=1000),
    )
    with pytest.raises(ValueError, match="Shared rendering prompt exceeds"):
        await judge(
            Transcript(
                transcript_id="overflow",
                messages=[ChatMessageUser(id="m", content="Hello")],
            )
        )
    assert not capture.requests
