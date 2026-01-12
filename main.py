import argparse
import asyncio
import os
import shutil
import subprocess

import kosong
from kosong.chat_provider import ChatProvider
from kosong.contrib.chat_provider.anthropic import Anthropic
from kosong.contrib.chat_provider.google_genai import GoogleGenAI
from kosong.contrib.chat_provider.openai_responses import OpenAIResponses
from kosong.message import Message
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from kosong.tooling.simple import SimpleToolset
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from typing_extensions import Optional

console = Console()
MODEL_CONFIGS = {
    "claude-opus-4-5": (
        Anthropic,
        {},
    ),
    "gemini-3-pro-preview": (
        GoogleGenAI,
        {},
    ),
    "gpt-5.1-codex-max": (
        OpenAIResponses,
        {},
    ),
    "gpt-5.2": (
        OpenAIResponses,
        {},
    ),
}

SYSTEM_PROMPT = """You are Laser, a coding agent for this repository.
You can use the terminal/shell to inspect and modify code and run commands.
Prefer fast search with ripgrep (e.g., `rg -n "symbol" .`), read files with
`cat`/`sed`, and list files with `rg --files` or `ls`. Avoid destructive
Respect `.gitignore`: avoid reading, searching, or modifying files/directories matched by
`.gitignore` unless the user explicitly asks.
actions (no `rm -rf`, no `git reset --hard`). After edits, check the diff
(`git diff`), re-open touched code to verify, optionally run relevant tests,
and then check status (`git status -sb`).
At the start of each task, check whether `AGENTS.md` exists at the repository
root. If present, read it first and follow its instructions. If it conflicts
with these instructions, prioritize these instructions.
Keep responses concise and focused on code changes."""


def parse_args():
    parser = argparse.ArgumentParser(
        prog="laser",
        description=(
            "Laser: a terminal-based coding agent you run inside a repository. "
            "It can inspect files, edit code, and run shell commands."
        ),
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        default="gpt-5.2",
        help="Model to use",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Default max tokens (used by Anthropic)",
    )
    parser.add_argument(
        "--cwd",
        type=str,
        default=None,
        help="Working directory to run the session from",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help=(
            "Run a single non-interactive prompt and exit. "
            "If omitted, starts an interactive session."
        ),
    )
    return parser.parse_args()


class UsageTracker:
    def __init__(self) -> None:
        self.input_other = 0
        self.input_cache_read = 0
        self.input_cache_creation = 0
        self.output = 0

    def add(self, usage) -> None:
        if usage is None:
            return
        self.input_other += usage.input_other
        self.input_cache_read += usage.input_cache_read
        self.input_cache_creation += usage.input_cache_creation
        self.output += usage.output

    @property
    def input_total(self) -> int:
        return self.input_other + self.input_cache_read + self.input_cache_creation

    @property
    def total(self) -> int:
        return self.input_total + self.output

    def render(self) -> str:
        input_total = self.input_total
        total = self.total
        cache_total = self.input_cache_read + self.input_cache_creation
        cache_pct = (cache_total / input_total * 100.0) if input_total else 0.0

        return (
            "Token usage (cumulative)\n"
            "\n"
            f"Input : {input_total:>8}  (other={self.input_other}, cache_read={self.input_cache_read}, cache_create={self.input_cache_creation}, cached={cache_pct:.1f}%)\n"
            f"Output: {self.output:>8}\n"
            "\n"
            f"Total : {total:>8}"
        )


def ensure_cwd(cwd: str | None) -> None:
    if cwd is None:
        return
    if not os.path.isdir(cwd):
        raise SystemExit(f"--cwd path does not exist or is not a directory: {cwd}")
    os.chdir(cwd)


def check_deps() -> None:
    missing: list[tuple[str, str]] = []

    if shutil.which("rg") is None:
        missing.append(
            ("ripgrep (rg)", "https://github.com/BurntSushi/ripgrep#installation")
        )

    if missing:
        console.print("Missing required dependencies:\n")
        for name, url in missing:
            console.print(f"- {name}: please install from {url}")
        raise SystemExit(1)


def render_banner(model: str) -> None:
    cwd = os.getcwd()
    console.print(
        Panel.fit(
            f"[bold]Laser[/bold]\n"
            f"- Model: [bright_cyan]{model}[/bright_cyan]\n"
            f"- Directory: [bright_cyan]{cwd}[/bright_cyan]\n"
            f"- Tip: type [bright_cyan]/quit[/bright_cyan] to exit\n"
            f"       type [bright_cyan]/reset[/bright_cyan] to clear history",
            border_style="bright_black",
        )
    )


def build_provider(model: str, max_tokens: int) -> ChatProvider:
    provider_cls, provider_kwargs = MODEL_CONFIGS[model]
    if provider_cls is Anthropic:
        provider_kwargs = {**provider_kwargs, "default_max_tokens": max_tokens}
    return provider_cls(model=model, **provider_kwargs)


def tool_messages(tool_results: list) -> list[Message]:
    return [
        Message(
            role="tool",
            content=tool_result.return_value.output,
            tool_call_id=tool_result.tool_call_id,
        )
        for tool_result in tool_results
    ]


class BashParams(BaseModel):
    command: str
    timeout: Optional[int] = 10
    short_user_facing_summary: str


class BashTool(CallableTool2[BashParams]):
    name = "bash"
    description = "Run bash commands"
    params = BashParams

    async def __call__(self, params: BashParams) -> ToolReturnValue:
        console.print(
            Padding(f"[bright_green]$: {params.short_user_facing_summary}", (0, 1))
        )

        result = subprocess.run(
            params.command,
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            timeout=params.timeout,
        )

        if result.returncode != 0:
            return ToolError(
                message="Bash command failed",
                brief="Command failed",
                output=result.stderr,
            )

        return ToolOk(output=result.stdout)


def cli() -> None:
    asyncio.run(main())


async def main():
    args = parse_args()
    ensure_cwd(args.cwd)

    if args.prompt is None:
        render_banner(args.model)

    chat_provider = build_provider(args.model, args.max_tokens)
    history = []
    toolset = SimpleToolset([BashTool()])
    usage_tracker = UsageTracker()

    if args.prompt is not None:
        history.append(Message(role="user", content=args.prompt))

        while True:
            result = await kosong.step(chat_provider, SYSTEM_PROMPT, toolset, history)
            usage_tracker.add(result.usage)
            history.append(result.message)
            tool_results = await result.tool_results()

            if len(tool_results) == 0:
                console.print(Padding(Markdown(result.message.extract_text()), 1))
                return

            history.extend(tool_messages(tool_results))

    while True:
        user_message = read_multiline_input(
            console,
            "[bold black on bright_cyan] laser [/]> ",
            "[bright_black]... [/]",
        )

        if user_message.startswith("!"):
            cmd = user_message[1:].strip()

            if not cmd:
                continue

            try:
                result = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    check=False,
                )
            except Exception as exc:  # noqa: BLE001
                console.print(Padding(f"[red]Command failed:[/] {exc}", 1))
                continue

            if result.stdout:
                console.print(Padding(result.stdout.rstrip("\n"), 1))
            if result.stderr:
                console.print(Padding(result.stderr.rstrip("\n"), 1))

            continue

        match user_message:
            case "/reset":
                history = []
                console.print(
                    Padding(
                        Panel.fit(
                            "Conversation reset. History cleared.",
                            title="/reset",
                            border_style="bright_black",
                        ),
                        1,
                    )
                )
                continue
            case "/usage":
                console.print(
                    Padding(
                        Panel.fit(
                            usage_tracker.render(),
                            title="/usage",
                            border_style="bright_black",
                        ),
                        1,
                    )
                )
                continue
            case "/quit":
                console.print("Goodbye [italic]sad computer making shutdown noises...")
                break
            case _:
                pass

        history.append(Message(role="user", content=user_message))

        while True:
            result = await kosong.step(chat_provider, SYSTEM_PROMPT, toolset, history)
            usage_tracker.add(result.usage)
            history.append(result.message)
            tool_results = await result.tool_results()

            if len(tool_results) == 0:
                console.print(Padding(Markdown(result.message.extract_text()), 1))
                break

            history.extend(tool_messages(tool_results))


def read_multiline_input(console: Console, prompt: str, continuation: str) -> str:
    first = console.input(prompt)

    if first.endswith("\\"):
        lines = [first[:-1].rstrip()]

        while True:
            next_line = console.input(continuation)
            if next_line.endswith("\\"):
                lines.append(next_line[:-1].rstrip())
                continue

            lines.append(next_line)
            return "\n".join(lines).strip()

    return first


if __name__ == "__main__":
    check_deps()
    asyncio.run(main())
