# AGENTS.md - Developer Guidelines & Environment Rules

## Environment & Tech Stack
- **OS:** Linux (Bash shell)
- **Language:** Python 3.10+
- **Virtual Environment:** Always use the local `.venv` directory (activate via `source .venv/bin/activate`).
- **Dependencies:** Install packages strictly via `pip` and update `requirements.txt` when necessary.
- **databases:** if for single user app, sqlite. Anything else, postgresql.

## Testing & Validation Rules
- **Skip Testing by default:**  Do not write test files or run unit tests. Can write tests when refactoring or make a replacing tools. Like postgresql for sqlite, python pymupdf for pdfplumber.
- **Manual Verification:** Rely strictly on direct script execution, CLI output checks, or dry-run validation logs to confirm code functionality.

## Python Execution Rules
- Always use the local virtual environment interpreter directly (.venv/bin/python on Linux/macOS or .venv\Scripts\python.exe on Windows) instead of running `source .venv/bin/activate`.
- Do not chain commands using `&&` for environment activation. Execute python/pip/manage.py commands using the direct path.o

## Summary & Daily File Logging (`YYYY-MM-DD_summary.md`)
Every new problem addressed and solved on a day (sudo or not) must be logged by appending a new section to the **end** of the dedicated daily summary file using the exact naming convention **`YYYY-MM-DD_summary.md`** (derived from the current system date) at the designated `logs/` directory. Replace `summary` with the actual subject that was worked on.

* **File Lifecycle:** 
  * If `YYYY-MM-DD_summary.md` does not exist for the current day, create it with a top-level date header.
  * If it already exists, append the new task entry directly to the end of the file.
* **Timestamp Requirement:** Each individual entry inside the daily file must include an exact timestamp using the format `HH:MM`.
* **Required Log Sections per Entry:**
  1. **Objective:** What system change, problem, or administrative goal was targeted.
  2. **Solution:** What was done to address it (include the script name/path when one was used).
  3. **Actions Performed:** Step-by-step breakdown of commands executed and their direct system impact.
  4. **Verification Output:** Proof of success (e.g., relevant snippets of `systemctl status`, `journalctl`, or package verification commands).
  5. **References Used:** Exactly 3 high-quality online references.
  6. **Optional steps:** Next steps to implement to improve. Security or convinence. Like reducing permissions or automation.
  7. **Possible problems** Things that can become problems. Problems what were not addressed. etc.

## Coding Style & Guardrails
- Write clean, modular, production-grade Python code with clear docstrings and type hints.
- functions should be independent and do one job only (mostly).
- Keep modifications scoped strictly to the requested feature or bug fix.
- Do not introduce external dependencies or databases unless explicitly authorized in the architecture specification.
- Never execute destructive git commands or commit changes automatically.
- Keep frontend minimal. HTMX. alpine.js to show loading or progress. dark mode. Prefer css over javascript in regards to web design.
- Create or update SUMMARY.md. Goal: for user too understand how it works and be able to create a simpler functional version on own. Give Summary of important steps taken and why. Provide timestamp
- in SUMMARY.md, also add sections: Suggestions (to improve). Possible problems.
- when you complete a task from `@TASKS.md`, mark it there as complete.



## Important reading 
- Read `@SPEC.md`, `@ROADMAP.md`, and `@PLAN.md`.  Implement **Phase number** of the roadmap, following PLAN.md as guide. Check `@TASKS.md` for the current sub-tasks to execute.
