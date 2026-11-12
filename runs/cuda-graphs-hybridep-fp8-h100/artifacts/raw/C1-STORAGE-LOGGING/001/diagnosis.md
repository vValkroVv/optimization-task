# C1 storage and logging diagnosis

## Acceptance and result

The bounded checks cover the affected logger path, preserved bytes and event sequences, current local mount capacity, and evidence consequences. C1 has confirmed output persistence failure; exact underlying NFS capacity/quota/backend cause remains unresolved. No source, GPU, cleanup, or process operations were performed.

## Pinned evidence

Repository base: `446431dfa191ef0300f116257ae918bc85e38c06`. SOURCE-SNAPSHOT/006 manifest SHA-256: `3a606c81d6b13692e422da04c333252620e504df9f1c4ba59c4bcf0c1a51f603`. See `evidence.json` for each artifact hash, byte offset, event and space check.

Native JSONL handler `gift/torchtitan/observability/structured_logger/jsonl_handler.py:164` subclasses `logging.FileHandler`, targeting attempt002/native/structured_logs/*.jsonl on NFS2. Its formatter assigns `seq_id` using an incrementing counter at line143. There is no custom duplicate/retry buffer in that handler. Span exit emits its end record in structured_logging.py:394; the optimizer span wraps clipping, optimizer.step and scheduler.step in trainer.py:793–815.

All eight JSONL files are parseable with no NUL bytes, but all lose seq1254–1276; six additionally lose1277 and contain an exact duplicate1278. Each retains1535 lines ending seq1557. The last common persisted record1253 is step17 post_dataloading_process_end at02:30:04.949539–.950828UTC. Ranks2/4 resume1277 at02:30:18.460743–.461334; other ranks resume1278 at02:30:26.745474–.862286. All contain25 optim_end records; only102 fwd_bwd_end records at ranks2/4 and103 at the other ranks, whose count includes duplicate1278. Required events are incomplete.

Rank0/6/7 stderr preserves logging flush failures for step18 fwd_bwd_end (durations8369.27/8339.08/8314.70ms); rank4 stderr also preserves Errno28 with a truncated stack. The Python logging frames are emit:1164 → flush:1144 → stream.flush, raising `OSError: [Errno 28] No space left on device`. Six stderr files contain NUL bytes (ranks0/1/3/5/6/7; first offsets3411/2606/2606/2628/2606/2899); supervisor.log contains17403 NUL bytes starting at offset24571. Native JSONLs contain none. The exact individual failed handler is not named by this truncated traceback; the structured-event call stack, simultaneous JSONL gaps and duplicate persistence strongly support failed JSONL writes, rather than skipped formatting or mere mislabeled steps. Supervisor.log:218–291 also preserves torchrun tailer failures in dst.write at02:34:47.644–48.647UTC. Torchrun redirects and supervisor output paths are under the same NFS2 attempt directory. Failed writes are caught by logging/tailer machinery, explaining why exit0 does not establish complete logs.

## Space and next action

The actual execution mount is /workspace-SR008.nfs2; the coordinator confirms all ranks execute on this same host through local standalone torchrun. At02:43:46UTC it had213GiB available, 100% rounded usage and9% inode use; a later byte check had225582514176 bytes available. Notebook /tmp is on separate overlay with1010819584000 bytes available. Current measurements establish this execution path has available bytes and inodes now, but do not establish capacity at02:30UTC or rule out backend allocation limits. The quota command is unavailable here; no quota was invented. No evidence supports deleting task checkpoints, and none were created in C1.

The current byte/inode capacity gate is open on the actual execution mount: it is not currently out of bytes or inodes. Prior ENOSPC remains unexplained at the backend level, so available capacity alone does not guarantee successful future writes. No-delete next action is to preserve logs and finish checking A008 output integrity, then let the coordinator decide any already-authorized next launch from current capacity and expected output size. No duplicate host check or new connection is needed. No retry is authorized by this diagnosis.

## Evidence impact

All eight timing observers preserve complete valid step21–25 windows closing26, with5242880 already-global valid tokens per rank and slowest-rank completed wall145.41152449091896s. These remain descriptive measurements with the documented timing scope. They do not restore missing structural evidence for the whole C1 run, prove every tensor/optimizer operation, or support a speedup claim. Missing optim_end17 cannot be interpreted as a skipped optimizer update, but logging causality alone cannot independently certify tensor semantics. The structural acceptance gate remains open.
