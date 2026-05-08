# Repository Guidelines

## Project Structure & Module Organization

`nmrkit/` contains the package source. Core data containers live in `nmrkit/core/`, processing functions in `nmrkit/processing/`, file readers in `nmrkit/io/`, utilities in `nmrkit/utils/`, and plotting helpers in `nmrkit/visualization/`. Processing workflows are grouped under `nmrkit/processing/workflows/`. Tests mirror the package layout under `tests/`, such as `tests/processing/test_ft.py` and `tests/core/test_data.py`. Build metadata and dependencies are defined in `pyproject.toml`; distribution output in `dist/` should not be edited manually.

## Build, Test, and Development Commands

- `poetry install`: install runtime and development dependencies.
- `poetry install --extras visualization`: include optional plotting support through `matplotlib`.
- `poetry run pytest`: run the full test suite.
- `poetry run pytest tests/processing/test_phase.py`: run a focused test file while developing.
- `poetry build`: build source and wheel distributions using the Poetry backend and dynamic git versioning.

The package requires Python 3.12 or newer.

## Coding Style & Naming Conventions

Follow PEP 8 with 4-space indentation. Use type hints for new public functions and concise docstrings matching the existing module style. Keep processing APIs function-oriented and domain-specific; prefer names like `ft`, `phase`, `zf`, or descriptive helpers over broad utility names. Test files should be named `test_*.py`, and test functions should describe behavior being checked. Use `black` to format code before committing.

## Testing Guidelines

Use `pytest` for all tests. Add or update tests whenever changing processing behavior, data validation, readers, or public APIs. Place tests in the matching area under `tests/`; for example, changes to `nmrkit/processing/window.py` belong in `tests/processing/test_window.py`. Prefer small numeric assertions that cover shape, dtype, metadata, and expected values for NMR data transformations.

## Commit & Pull Request Guidelines

Use short imperative messages with a type prefix such as `feat:`, `fix:`, `test:`, or `docs:`.

Pull requests should include a clear summary, the motivation for the change, relevant issue links, and the tests run. Include screenshots or generated plots when visualization behavior changes. Keep PRs scoped to one feature or fix, and avoid unrelated formatting churn.

## Security & Configuration Tips

Do not commit raw proprietary NMR datasets, credentials, or local environment files.