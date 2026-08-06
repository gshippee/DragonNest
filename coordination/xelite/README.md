# X Elite file-mediated coordination

This directory is the intentionally slow Agent-to-Agent mailbox for the
physical Snapdragon X Elite laptop.

- `AGENT_PROMPT.md` is the current instruction for the laptop-side agent.
- `STATUS.json` is the only response file it should edit and commit initially.
- Full logs and `dragonnest-xelite-proof.json` stay outside Git under `%TEMP%`.
- `STATUS.json` records only sanitized facts and the proof file's SHA-256.

The laptop agent should create and push a response branch named
`codex/xelite-response`. Shubham can then relay that branch name to the primary
agent. Never place tokens, credentials, SDK/runtime binaries, model bundles,
licenses, certificates, raw prompts containing private data, or verbose system
logs in this mailbox.
