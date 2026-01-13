# Repository Guidelines

## Project Structure & Module Organization
- `main.py`: streaming ASR entrypoint with CLI flags (`--mic`, `--audio_file`, `--benchmark`).
- `play_npy_audio.py`: utility to convert `.npy` audio to `.wav` for playback.
- `outputs/`: optional location for generated artifacts (e.g., converted WAVs); recordings default to `recording_YYYYMMDD_HHMMSS.npy` in repo root.
- `debug_audio.npy` / `debug_audio.wav`: sample assets for local verification.
- `pyproject.toml` / `uv.lock`: Python dependencies and lockfile.

## Build, Test, and Development Commands
- `uv sync`: install locked dependencies (preferred).
- `uv run python main.py --mic`: run realtime recognition via `uv`.
- `uv run python main.py --audio_file debug_audio.npy --benchmark "基准文本"`: stream a saved audio file and auto-validate text similarity.
- `uv run python play_npy_audio.py recording_*.npy --output outputs/example.wav --play`: convert to WAV and play; requires `scipy`.
- `python main.py --mic`: run realtime recognition from the default microphone.
- `python main.py --audio_file debug_audio.npy --benchmark "基准文本"`: stream a saved audio file and auto-validate text similarity.
- `python play_npy_audio.py recording_*.npy --output outputs/example.wav --play`: convert to WAV and play; requires `scipy`.

## Coding Style & Naming Conventions
- Python with 4-space indentation and PEP 8 naming (`snake_case`, `UPPER_SNAKE_CASE` for constants).
- Keep CLI flags descriptive and long-form (e.g., `--audio_file`, `--benchmark`).
- Preserve current user-facing text language and emoji usage for consistency in terminal output.

## Testing Guidelines
- No automated tests yet; validate changes by running `main.py` with `--mic` or `--audio_file`.
- If introducing tests, add `tests/` with `test_*.py` and document the runner (e.g., `pytest`).

## Commit & Pull Request Guidelines
- Use Conventional Commits; example: `feat: add real-time streaming ASR flow`.
- PRs should include: a short description, commands run, and any relevant audio artifacts or transcripts when behavior changes.
- Note new dependencies, model revisions, or changes that affect model download or microphone permissions.

## Configuration & Runtime Notes
- First run may download models via FunASR; network access is required.
- Microphone input depends on OS permissions; document any OS-specific setup in the PR when needed.
