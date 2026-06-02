# BoidForge — Setup & Git Workflow

Get a clean machine to the point where `pytest` is green and the package
imports, then set up the branch/tag workflow.

---

## 1. Prerequisites

| Tool         | Version  | Why                                              |
|--------------|----------|--------------------------------------------------|
| Python       | ≥ 3.10   | Runtime + build host                             |
| Git          | any      | Version control                                  |
| C compiler   | per-OS   | Builds the `boidforge._native` extension         |
| CMake        | ≥ 3.18   | Native build (auto-provisioned by the backend)   |

C compiler per OS:

- **Windows** — *Build Tools for Visual Studio* with the **Desktop development
  with C++** workload (provides MSVC `cl.exe`).
- **macOS** — Xcode Command Line Tools: `xcode-select --install`.
- **Linux** — `sudo apt install build-essential python3-dev`.

CMake and Ninja are fetched automatically by `scikit-build-core` during the
build, so a manual CMake install is optional.

---

## 2. Install the toolchain (Windows)

1. **Python** — install from <https://www.python.org/downloads/>, ticking
   **"Add python.exe to PATH"**. Verify in a new terminal:

   ```powershell
   python --version
   ```

2. **Git** — install from <https://git-scm.com/download/win>. Verify:

   ```powershell
   git --version
   ```

3. **MSVC C++ build tools** — download *Build Tools for Visual Studio* from
   <https://visualstudio.microsoft.com/visual-cpp-build-tools/>, run the
   installer, and select **Desktop development with C++**. Reboot the terminal
   afterward so the compiler is on PATH.

---

## 3. Create the virtual environment

```powershell
cd D:\Codes\boid_forge
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation ("running scripts is disabled"), allow it for
the current session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
```

(Command Prompt instead of PowerShell: activate with `.venv\Scripts\activate.bat`.)

---

## 4. Install BoidForge (editable)

This compiles the C extension and installs the dev tools:

```powershell
pip install -e ".[dev]"
```

- For everything (visualization + benchmark extras): `pip install -e ".[all]"`.
  The `viz` extra pulls ModernGL/pyglet, which need a GPU/display.
- **No C compiler yet?** Install without the native extension — the package and
  the L1/L2 Python solvers do not require it:

  ```powershell
  pip install -e ".[dev]" --config-settings=cmake.define.BOIDFORGE_BUILD_EXTENSION=OFF
  ```

---

## 5. Verify

```powershell
pytest -q
python -c "import boidforge; print(boidforge.__version__)"
```

Expected at this skeleton stage: **16 passed, 6 xfailed** (the 6 xfails are the
not-yet-implemented solver/IO contracts — that is correct, not an error), and
the version prints `0.1.0`.

Optional quality gates (clean once code lands):

```powershell
ruff check .
mypy --strict src/boidforge
```

---

## 6. Git workflow

### One-time: first commit, remote, and branches

`main` already exists locally. Create an **empty** GitHub repo first (no README,
.gitignore, or license — those already exist here), then:

```powershell
cd D:\Codes\boid_forge

# If git complains about a stale lock from an interrupted op:
#   del .git\index.lock

git branch -M main                       # ensure the branch is named main
git add .
git commit -m "chore: scaffold BoidForge — architecture, skeleton, build, tests"

# Replace <you> with your GitHub username (HTTPS shown; SSH also fine)
git remote add origin https://github.com/vcaries/boid_forge.git
git push -u origin main

# Long-lived dev branch off main
git checkout -b dev
git push -u origin dev
```

(Alternatively, with the GitHub CLI: `gh repo create boid_forge --private
--source=. --remote=origin --push`.)

### Day-to-day: develop on `dev`

```powershell
git checkout dev
# ... implement, e.g. the L1 solver ...
git add -A
git commit -m "feat(solver): implement L1 naive reference"
git push
```

Use clear, conventional commit prefixes: `feat`, `fix`, `perf`, `refactor`,
`test`, `docs`, `chore`.

### At each milestone: merge to `main` and tag a version

```powershell
git checkout main
git merge --no-ff dev -m "Milestone: solver L1/L2/L3 complete"

# Annotated, semver-style tag
git tag -a v0.1.0 -m "v0.1.0 — solver reference + spatial hash + native kernel"

git push origin main --follow-tags    # pushes the merge and the tag

# Resync dev so it includes the merge commit
git checkout dev
git merge main
git push
```

`--no-ff` forces a merge commit so each milestone is a visible point in
`main`'s history. Version tags follow [SemVer](https://semver.org/):
`vMAJOR.MINOR.PATCH` — bump MINOR for each feature milestone (`v0.2.0`,
`v0.3.0`, …), PATCH for fixes, MAJOR at the first stable release (`v1.0.0`).
