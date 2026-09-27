# OSM-46A — real Windows pilot execution pack

Status: **IMPLEMENTED / CI CONTRACT CERTIFICATION PENDING**

## Goal

Turn the certified OSM-42D through OSM-45C contracts into one reproducible,
restart-safe Windows pilot procedure with no hand-editing of intermediate JSON.

OSM-46A does not execute Qwen 80B in CI. It packages and validates the exact
commands that the designated Windows host must execute.

## Two artifacts

### Immutable execution pack

`build_qwen_windows_pilot_pack.py` binds:

- OSM-42B machine authorization;
- OSM-41A execution manifest;
- OSM-41B operator state;
- OSM-41D command plan;
- checkpoint and tokenizer directories;
- OSM-44A chat-template reference;
- prompt and structured-message files;
- native probes for OSM-43A/B/C/D and OSM-44C;
- prompt/generation/cache bounds;
- SHA256 for every immutable input.

Every command is stored as an argv array. No production phase uses a shell
command string.

### Restart-safe runner

`run_qwen_windows_pilot_pack.py` executes, in order:

1. OSM-42D Windows conversion certification;
2. OSM-43A runtime load;
3. OSM-43B first token;
4. OSM-43C text first token;
5. OSM-43D bounded generation;
6. OSM-44C chat-formatted bounded generation;
7. OSM-45A cold performance instrumentation;
8. OSM-45B prefill/decode latency certification;
9. OSM-45C cache-phase attribution.

After each phase the runner validates the expected evidence stage before
committing progress to its state file. A rerun skips only phases whose output
still validates.

## Safety and claim boundary

The pack itself may claim only that the real Windows pilot is planned and that
its evidence chain is deterministic.

CI may certify:

- argv-only execution;
- phase ordering;
- restart-safe resume;
- evidence-stage validation;
- input hash binding;
- tamper rejection.

CI may **not** claim:

- the full official checkpoint was converted on the user's machine;
- real 80B inference ran;
- measured tokens/s are representative;
- a performance target was met.

Only a completed state file produced on the designated Windows host after all
nine real phases may claim `real_windows_pilot_executed=true`.

## Next

**OSM-46B — Windows pilot evidence seal**

After the real designated-host run, bind the final OSM-46A state, all nine
evidence files, host identity, package digest, and performance/cache summaries
into one immutable pilot certification.
