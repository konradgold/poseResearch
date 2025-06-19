See https://docs.astral.sh/uv/guides/projects/ for any further information.

---

# Format

Check whether all files are properly formatted:

```bash
black . --check
```

Run without the `--check` flag to format the files.

If you add a directory to root that should not be formatted, add it to the exclude pattern in the [pyproject.toml](pyproject.toml).

# Lint

Check whether any file needs linting:

```bash
ruff check
```

If you add a directory to root that ruff should not be check, add it to the exclude pattern in the [pyproject.toml](pyproject.toml).
