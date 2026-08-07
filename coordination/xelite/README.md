# X Elite file-mediated coordination

This directory began as the file-mediated mailbox for the physical Snapdragon
X Elite laptop. The physical worker is now verified and the mailbox is kept as
a sanitized audit trail.

- `AGENT_PROMPT.md` is now the no-reasoning operator handoff for the worker.
- `STATUS.json` summarizes the completed worker and PersonaCare two-request
  stage proof.
- `docs/results/xelite_worker_status.md` contains the detailed sanitized proof.
- Full logs and `dragonnest-xelite-proof.json` stay outside Git under `%TEMP%`.
- `STATUS.json` records only sanitized facts and the proof file's SHA-256.

No new laptop-side response branch or LLM session is expected for normal
operation. Never place tokens, credentials, SDK/runtime binaries, model
bundles, licenses, certificates, raw prompts containing private data, or
verbose system logs in this mailbox.
