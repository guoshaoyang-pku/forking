# Experiment-log active/history audit · 2026-09-13

This is a machine-assisted audit of status-like wording in `docs/experiment-log.md`. It does not rewrite historical entries. Current coordination status is taken from `overnight-dispatch-manifest.json` and `overnight-queue-20260913.md`; a marker in the experiment log is not evidence that a run is currently active.

## Classification

| line(s) | wording | classification | authority / action |
|---:|---|---|---|
| 4, 111, 113 | planned / running / done lifecycle | policy text | keep; applies to registration semantics |
| 107, 2151 | order-5 Transformer planned / waiting for GPU confirmation | open historical design | keep as unstarted; do not schedule without a new registration |
| 255 | “other Agent running” | historical section guidance | keep; not a current process claim |
| 586, 672, 742 | historical shard-scan note;待核 snapshot | superseded history | keep provenance; current interpretation is §40 and S1 v5 |
| 950, 1160, 1428, 3153, 3850 | planned registration/proposal text | historical plan or future proposal | keep; queue decides whether it is active |
| 2722, 2799, 3096, 3232, 3939, 3947 | U-shape / wrap-around wording | erratum/provenance | keep; corrected claim is monotone real-length segment plus replay/pass interpretation |
| 3373, 3983 | running → done transition snapshots | resolved historical transition | keep; later backfill is authoritative |

## Result

The scan found 23 status-like lines. All non-policy markers are either explicitly historical, superseded, or open design records. No line in the active coordination surface promotes a historical `planned`/`running` snapshot to a current execution. The machine-readable queue contains 27 unique IDs (`D1..D27`) and remains the current source for overnight status.

Validation command:

```bash
python3 docs/coordination/check_navigation_links.py
```

Current result: `checked=31 missing=0`.
