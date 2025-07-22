See https://docs.astral.sh/uv/guides/projects/ for any further information.

---

# Run this project

```bash
pip install -e .
```

Examples from [poseResearch/example_usage.py](poseResearch/example_usage.py) now work out of the box if they do not use the cloned subrepositories. If you need them use uv to create a virtual environment:

```bash
uv venv
.venv/Scripts/activate
```

Some estimation classes need additional setup like to download checkpoints. Read the respective class descriptions.

# Development

## Format

Check whether all files are properly formatted:

```bash
black . --check
```

> Tip: Run without the `--check` flag to format the files.

If you add a directory to root that should not be formatted, add it to the exclude pattern in the [pyproject.toml](pyproject.toml).

## Lint

Check whether any file needs linting:

```bash
ruff check
```

> Tip: Running with the `--fix` flag will fix some linting errors.

If you add a directory to root that ruff should not be check, add it to the exclude pattern in the [pyproject.toml](pyproject.toml).
