# RAG with LangChain

Python **3.9+**. Dependencies are listed in `pyproject.toml` (and mirrored in `requirements.txt`).

## Install with uv (recommended)

[Install uv](https://docs.astral.sh/uv/getting-started/installation/) if you do not have it.

From the project root:

```bash
uv sync
```

That creates `.venv` and installs everything from `pyproject.toml` / `uv.lock`.

Activate the environment:

```bash
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows
```

Jupyter notebooks: select the `.venv` kernel (`ipykernel` is already a project dependency).

Install extra packages later:

```bash
uv add <package>
# or
uv add -r requirements.txt
```

## Install with pip

```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS / Linux
# .venv\Scripts\activate    # Windows

pip install -U pip
pip install -r requirements.txt
pip install ipykernel       # needed for notebooks; not listed in requirements.txt
```

`requirements.txt` has no version pins. For a locked, reproducible install, use `uv sync` instead.
