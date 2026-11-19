# Instructions for coding agents (Codex, ChatGPT and others)

Follow `CLAUDE.md` in this folder: it holds the project rules. Summary of the hard rules:

- Read `docs/CONTRACTS.md` before writing code; it is the agreement between the four zones. Do not change it without the team.
- Work only in the files of your zone (table in `docs/TASKS.md`).
- Core package `auditgym/`: Python 3.11 standard library only. Tests use `unittest`.
- Randomness only through `auditgym.rng.Rng`, passed as an argument. No `random`, `uuid`, clock, or iteration over unordered sets.
- Money is `int` in minor units; rates and arithmetic use `Decimal`; no `float`.
- Hidden truth, provenance and the seed never appear in observations, `info`, error messages, paths or trajectories.
- Before every push: `python3 -m auditgym check` must end with `CHECK: PASSED`.
- Code, identifiers and commits in English; README and design note in Russian.
