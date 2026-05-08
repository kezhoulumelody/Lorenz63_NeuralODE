# Contributing

Thanks for helping improve this project.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ".[dev]"
```

## Guidelines

- Keep changes focused and easy to review.
- Add tests for new behavior when practical.
- Prefer clear names and small functions.
- Document experiment assumptions, datasets, and generated artifacts.

## Pull Requests

Before opening a pull request:

```bash
pytest
```

Include a short summary of the change, any relevant results, and notes about limitations or follow-up work.
