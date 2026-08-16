# GA3BAD Ollama model-switch reliability update

Date: 2026-08-15

## Summary

This update fixes the model-switching failure in which GA3BAD could display a
new Ollama model in the terminal, `/live`, and `/advanced-tracing` while later
reporting a misleading local-model load error. Model identity, Ollama inventory,
provider probing, first response, and GA3BAD contract validation are now tracked
as separate facts.

The implementation was validated by switching one durable session from
`gemma4:e4b` to the exact Ollama cloud alias `gpt-oss:120b-cloud`, creating a
one-file Python program, and executing it successfully.

## What changed

- Ollama selection requires an exact catalog name. If `gpt-oss:120b` is absent
  but `gpt-oss:120b-cloud` exists, GA3BAD rejects the plain name and suggests the
  installed alias instead of fabricating a selectable descriptor.
- Model switches are transactional. GA3BAD probes the candidate before
  replacing the current provider, so a failed switch preserves the working
  model and saved workflow.
- Runtime state and both live views report five independent lifecycle facts:
  selected, discovered, probe passed, response passed, and contract verified.
- Recovery diagnosis distinguishes runner/network/model-load failures from
  empty responses, tool-contract failures, and typed-return validation errors.
- A required native tool stage may retry once through the governed constrained
  JSON action adapter when Ollama returns no native tool call.
- Architecture payloads accept transport-only wrappers and common field aliases
  without inventing missing model content.
- Ultra validated-response events now update the same model lifecycle used by
  the normal runtime.
- Approval-bound commands such as `python hello.py` are accepted by the tester
  allowlist while shell composition remains rejected.
- Change-set reconciliation can no longer let majority consensus overwrite a
  mandatory reviewer rejection.

## Live validation

Validation workspace:
`projects/ga3bad-model-switch-hello-final`

Session:
`workspace-880b65411a78`

Sequence:

1. Created the empty session with `gemma4:e4b`.
2. Discovered and preflighted `gpt-oss:120b-cloud` through Ollama 0.32.9.
3. Switched the same saved session transactionally.
4. Confirmed that response and contract states reset to `not_run` immediately
   after the switch.
5. Submitted: `Create hello.py that prints exactly Hello from GA3BAD! and verify
   it by running python hello.py. Keep the project to this one file.`
6. Approved only the exact bounded command `python hello.py`.
7. GA3BAD recorded `exit code: 0` and stdout `Hello from GA3BAD!`.
8. Independently reran the program and received the same output with exit 0.

Final persisted state:

| Fact | Result |
| --- | --- |
| Model | `gpt-oss:120b-cloud` |
| Execution class | cloud |
| Inventory | discovered |
| Capability probe | passed |
| Model response | passed |
| GA3BAD contract | verified at route |
| Pending semantic turn | none |
| Workflow run state | idle |
| Product files | only `hello.py` |
| Program result | exit 0, exact expected stdout |

## Problems found during validation

- `gemma4:e4b` was very slow in the recursive workflow and failed its first
  master plan because a verification module omitted required `write_paths`.
  The new diagnosis correctly reported a master-plan contract problem and
  preserved the checkpoint instead of calling it a local-model load failure.
- The cloud model's first semantic-router request had a transient typed provider
  error. Retry 1/3 succeeded without losing the selected model or session.
- A direct Python verification command was missing from the tester allowlist.
  This was fixed and regression-tested.
- Ultra responses initially did not update the shared lifecycle snapshot. This
  was fixed and regression-tested.
- Change-set reconciliation could approve majority consensus even when a
  required reviewer rejected the checkpoint. This was fixed so all mandatory
  reviews remain authoritative.
- During the final successful Action run, the cloud model made three malformed
  output-publishing tool calls. The bounded correction budget stopped the loop,
  and GA3BAD's harness still published the evidence-backed Output artifact. The
  task and exact runtime verification completed, but the contradictory terminal
  sentence saying delivery stopped before showing the successful Delivery block
  remains a presentation-quality follow-up.

## Automated verification

- Focused post-fix model/Ultra regression gate: 26 passed.
- Python compilation: passed.
- Earlier model/provider/catalog gate: 51 passed.
- Earlier runtime gate: 116 passed.
- Live bounded model smoke for `gpt-oss:120b-cloud`: passed through native tool
  transport.

## Reproduction

Run the bounded provider/contract smoke test:

```powershell
python scripts/model_smoke.py --model gpt-oss:120b-cloud
```

For an interactive switch, open GA3BAD, use F3 or Runtime / Model in Settings,
select the exact `gpt-oss:120b-cloud` catalog row, and confirm that inventory and
probe pass before the first response/contract states change. Then submit a small
task and inspect `/live` or `/advanced-tracing` after the first validated stage.
