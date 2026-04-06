# Contributing

AerisPlane is in early development. Issues and pull requests are welcome.

## Setup

```bash
git clone https://github.com/KasparStepan/AerisPlane.git
cd AerisPlane
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/
pytest tests/test_aero/   # aerodynamics only
```

## Linting

```bash
ruff check src/
```

## Building the docs locally

```bash
pip install -e ".[docs]"
mkdocs serve
```

Open `http://127.0.0.1:8000` to preview the site.

## Branch naming

- `feature/<name>` — new functionality
- `fix/<name>` — bug fixes
- `docs/<name>` — documentation only

Submit a pull request against `main`. Please include a brief description of what
changed and why.
