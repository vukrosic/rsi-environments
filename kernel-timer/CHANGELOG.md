# kernel-timer changelog

## v5.1 (2026-10-16)

- Warm-up window on an independent `os.urandom` seed (v5: derived from the timed seed; advisory "predictable warmup inputs").
- Kernel reports validated: exactly 4 finite fingerprints and a finite overhead >= 0, else fault candidate (advisory "unchecked fingerprint length").
- Non-finite speedup fails as fault infrastructure.
- Output memory and jail copies tracked and removed in `finally`; fallback output file opened `O_CREAT | O_EXCL`, mode 0600.
