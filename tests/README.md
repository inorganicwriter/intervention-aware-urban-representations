# Tests

- `unit/`: deterministic tests without network access.
- `integration/`: cross-module or data-contract tests.
- `fixtures/`: static test inputs such as saved HTML.

Tests must not access live external sites. Network acquisition is exercised through
saved fixtures or explicit operator-run commands under `scripts/collection/`.
