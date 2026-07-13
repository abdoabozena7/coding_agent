# Schema v4 → v5 migration note

Opening an existing workspace upgrades `.coding-agent/state.db` from v3 to v4 in one SQLite `BEGIN IMMEDIATE` transaction. The migration is additive: legacy goals, plans, approvals, tasks, Ultra runs, artifacts, traces, and Project Brain records are not rewritten or deleted.

The executable-Chat release then applies schema v5 in another additive transaction.
It adds `chat_messages`, `chat_artifacts`, `session_actions`, and
`managed_resources`. Generated-code bodies are zlib-compressed and SHA-256
addressed; existing Goal/Ultra rows remain unchanged.

Version 4 adds `workflow_sessions`, `agent_registry`, `quality_policies`, `change_sets`, `mutation_ledger`, `quality_findings`, and `quality_cycles`, plus their indexes. Version 5 adds the ordinary-Chat execution records above. `CREATE TABLE/INDEX IF NOT EXISTS` makes both migrations safe to invoke again after a completed upgrade. SQLite rolls the transaction back if any statement fails; `PRAGMA user_version` changes only at commit.

After restart, persisted findings, Change Sets, cycles, and reports remain available. Sleep returns to off and its profile returns to standard because Full Docker access must be selected again for each process. No provider or model is changed automatically.

Before upgrading, normal filesystem backups of `.coding-agent/state.db` remain appropriate. A database produced by v4 is intentionally rejected by older binaries that only understand v3.
