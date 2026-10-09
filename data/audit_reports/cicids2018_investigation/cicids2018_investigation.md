# CIC-IDS2018 Stage 6A.1 investigation

## Scope and provenance

- Source CSV: `C:\Users\Acharya N\Desktop\SOLO-version\FPGuard-AI\data\processed\cicids2018\cicids2018_audit_ready.csv`
- SHA-256: `aa7ace995ac68a15cfd5db91af76040060967d2a7e3e0cb7bde85479799b107c`
- Rows / columns: 613,071 / 80
- Diagnostic only: no labels changed, no splits created, no model trained.

## Feature-conflict investigation

Stage 6A groups rows by **77 exact parsed feature columns**:
all columns except `label`, plus `dst_port` and `timestamp`, which its name-based
review flags. It uses pandas `groupby(..., dropna=False)` and does not round.
Independent reproduction found 7,848 conflicting
groups involving 72,483 rows: benign
53,745; attack
18,738.

- Exact full-row duplicates (including label): 6,089
  excess rows in 5,814 groups.
- Identical values across all 79 feature columns but different labels:
  6,601 groups,
  13,204 rows.
- Conflict groups with variation in excluded columns:
  `{"dst_port+timestamp": {"group_count": 2134, "rows": 48251}, "no_excluded_field_variation": {"group_count": 312, "rows": 624}, "timestamp": {"group_count": 5402, "rows": 23608}}`.
- Rounded-feature causation remains unresolved: Stage 6A did no rounding, but
  upstream extraction precision is not established by these files.

Representative examples are in the JSON as SHA-256 feature
fingerprints, source CSV record positions, label counts, and excluded-field
variation counts. Raw feature rows are not copied into the report. These
conflicts are observations, not proof of label error; do not drop or relabel
them without provenance review.

## Leakage and capture-context review

- `timestamp` is an actual timestamp field; 0 parse
  failures across 613,071 rows and
  1 date(s) were observed.
  Only one date (2018-02-28) is present; a cross-day temporal holdout is not supported by this file.
- `dst_port` is a destination service port with
  13650 distinct observed
  values. Stage 6A flags it because `dst` matches a broad identifier-name
  heuristic; that heuristic does not prove it is a unique identifier.
- `protocol` counts: `{"6": 385716, "17": 216615, "0": 10740}`.
- Stage 6A name heuristics flagged
  `{"possible_identifiers": ["dst_port"], "possible_timestamps": ["timestamp"], "possible_attack_category_columns": [], "possible_capture_context_columns": [], "possible_target_or_post_event_columns": []}`.
  No source/destination IP or flow-ID, attack-category, capture-session, or
  target/post-event columns appear by these checks. This is schema evidence,
  not proof that flow statistics are free of scenario or collection artifacts.

Recommended: exclude raw `timestamp` from predictive inputs; consider parsed
capture-day solely for chronology/grouping if trustworthy. Treat `dst_port`
and `protocol` as candidate behavioral predictors only after documenting
deployment relevance and checking scenario/capture association. Group exact
duplicates or identical predictive-feature vectors in one partition. Do not
group solely on port.

## Preprocessing and evaluation recommendations

- Full-file audit observed 4,041 missing cells;
  missing feature details and infinities are listed in the JSON.
- Convert infinities to missing, then fit imputers on training rows only.
- Constant features observed on the whole file:
  `bwd_psh_flags, bwd_urg_flags, fwd_byts/b_avg, fwd_pkts/b_avg, fwd_blk_rate_avg, bwd_byts/b_avg, bwd_pkts/b_avg, bwd_blk_rate_avg`.
  If dropping them, make that decision in training-fitted preprocessing only.
- The recommended primary split, if proceeding, keeps exact groups made from
  **all 79 non-label columns** indivisible across train/validation/test and
  balances class counts only at the group level. Because groups can contain
  both labels, ordinary row-level stratification is inappropriate; verify class
  presence and actual prevalence after assignment.
- Timestamp evidence supports no cross-day holdout. Grouping all 79 fields
  prevents exact-feature duplicates crossing partitions, not near-duplicates,
  related flows, or all capture artifacts. No IP/flow/session identifiers exist
  for stronger grouping. A within-day chronological holdout would assess only
  later records from this same capture context—not another day, network, or
  attack family.
- Keep conflicts intact for investigation; report their effect and limitations
  in any later experiment. Use train-only preprocessing, validation-only
  selection, and untouched test evaluation if Stage 6B proceeds.

## Status

This report is diagnostic evidence only. Upstream rounding, capture/session
independence, and the cause of conflicting labels remain unresolved. No Stage
6B run or model-performance claim is included.
