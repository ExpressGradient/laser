# Laser

Laser is a small, terminal-based coding agent you run inside a repository. It can:

- read and edit files in the repo
- run shell commands (tests, formatters, build steps)
- iterate with you on changes, keeping the workflow repo-first

It’s built on [`kosong`](https://pypi.org/project/kosong/) and supports multiple model providers.

## Quick start (no clone required)

### 1) Install `uv`

If you don’t have it yet, install `uv` (recommended Python package manager):

- https://docs.astral.sh/uv/

### 2) Run Laser from GitHub

You can run Laser directly from this repo without downloading it:

```bash
uvx --from git+https://github.com/ExpressGradient/laser laser
```

Useful flags:

```bash
uvx --from git+https://github.com/ExpressGradient/laser laser --help
uvx --from git+https://github.com/ExpressGradient/laser laser --model gpt-5.2
uvx --from git+https://github.com/ExpressGradient/laser laser --cwd /path/to/your/repo
uvx --from git+https://github.com/ExpressGradient/laser laser --max-tokens 4096
```

Notes:

- `--cwd` changes the working directory Laser operates in (where it reads/writes files and runs commands).

## Using Laser effectively

### Basic workflow

1. Start Laser in your repo.
2. Describe what you want to change.
3. Laser will inspect the repo, make edits, and run commands as needed.
4. Review diffs and iterate.

Laser is intentionally conservative:

- it prefers `rg` for search
- it avoids destructive commands (e.g. `rm -rf`, `git reset --hard`)
- it checks diffs and file contents after edits

### Multi-line input

If your message ends with a trailing backslash (`\`), Laser will continue reading lines until you send a line without `\`.

Example:

```text
Write a script that:\
- scans all Python files\
- prints unused imports\
- and suggests fixes
```

### Repository instructions (`AGENTS.md`)

If your repository has an `AGENTS.md` at its root, Laser will read it at the start of each task and follow those repo-specific rules (unless they conflict with Laser’s own safety/workflow rules).

This is useful for documenting things like:

- how to run tests
- formatting/linting commands
- code style conventions
- release steps

## Models and providers

Laser supports multiple model backends (configured in `main.py`). Select one with `--model`.

```bash
uvx --from git+https://github.com/ExpressGradient/laser laser --model gpt-5.1-codex-max
```

If a provider requires environment variables (API keys), set them before running. Common examples:

- OpenAI: `OPENAI_API_KEY`
- Anthropic: `ANTHROPIC_API_KEY`
- Google: `GOOGLE_API_KEY`

(Exact names depend on the provider SDK/config; check provider docs.)

## Development

If you *are* working on Laser itself, run it locally:

```bash
uv run python main.py
```

Or via the console script from your checkout:

```bash
uvx --from . laser
```

## Troubleshooting

### Token usage

Type:

```text
/usage
```

To print cumulative token usage for the current session.

### Missing ripgrep (`rg`)

Laser requires [`ripgrep`](https://github.com/BurntSushi/ripgrep) (the `rg` command). If it's not installed, Laser will exit at startup with a message pointing you to install it.

### Shell commands fail

Laser surfaces stderr from failed commands. If something fails, try:

- re-running the exact command manually
- checking that required tools are installed
- ensuring you’re in the correct directory (`--cwd`)

### The agent is operating in the wrong directory

Run with:

```bash
uvx --from git+https://github.com/ExpressGradient/laser laser --cwd /path/to/repo
```
