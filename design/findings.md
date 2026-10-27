# Findings

A finding is a standing question about a project's transcripts. It runs as a scanner,
its results become reviewable evidence, and its answer is rebuilt from the evidence
that has been confirmed. User documentation: `docs/findings.qmd`.

## Concepts

- **Finding**: `findings/<slug>/finding.yaml` holds the slug, title, prompt, optional
  labels and `prompt_sha` (a hash of prompt and labels; changing either starts a new
  coverage version).
- **Finding scanner**: each finding runs as the scanner `finding_<slug>`, which asks the
  prompt of every transcript segment and returns items with a verbatim quote, a reason
  citing `[M<n>]` and an optional label.
- **Exhibit**: one cited message, identified by message id. A message cited again by a
  later scan gains a **sighting** rather than becoming a second exhibit.
- **State**: `proposed` (cited, not yet decided), `confirmed` or `rejected`.
- **Verdict ledger**: `verdicts.jsonl`, an append-only log of human and judge
  decisions. The latest human row wins; otherwise the latest judge row applies.
- **Exhibit judge**: decides pending exhibits, using the finding's human decisions
  as worked examples. Each run is logged in `judge_runs.jsonl`.
- **Agreement lines**: per scan, how many human-confirmed and human-rejected messages
  it cited; for the judge, how many of its decisions humans reviewed and kept.
- **Answer**: `answer.md`, a computed section plus a model-written synthesis built only
  from confirmed exhibits.

## Flow

1. `scout findings register` writes the finding and, if absent, the project nonce.
2. A scan runs the findings it names (`scout scan --findings a,b|all`, or
   `scan(findings=...)`); `add_finding_scanners` adds their scanners to the job. All
   finding scanners share one `ScannerRendering` (message roles, segment budget,
   nonce) and one answer schema, so their transcript prefix is byte-identical and can
   be cached by the provider.
3. Results land in the project's scans directory like any scanner's. When the scan
   finishes, `finalize_findings` refreshes each named finding's answer, including scans
   that ended with errors.
4. Derivation (`derive_finding`) reads every project scan that ran the finding:
   - each item's quote is verified against the message it cites; if it is not there,
     the transcript is searched and the item relocated, or marked `unlocated`;
   - items are grouped by message id into exhibits with their sightings;
   - the ledger's decisions are applied.

   A completed scan never changes, so its contribution is stored in
   `findings/.derived/` and read from there afterwards.
5. Review appends human rows: the Findings tab, or `scout findings review`.
6. `scout findings judge` (or the tab's judge button) decides pending exhibits and
   appends judge rows; a human decision is never re-judged.
7. `refresh_answer` rebuilds the computed section from the derived view and
   regenerates the synthesis only when the evidence digest (confirmed exhibits, their
   labels and quotes, the prompt and the synthesis instructions) has changed.
   Verdict clicks in the tab keep the previous synthesis, marked out of date, until
   the answer is regenerated.

## Invariants

- A finding's exhibits come only from its own scanner, plus human decisions on
  messages it did not cite. Other scanners' citations are not exhibits.
- A scan runs a finding only when it names it.
- Human decisions outrank judge decisions; a rejected message stays rejected however
  often later scans cite it.
- Evidence is exactly the confirmed exhibits, whether a human or the judge confirmed
  them. Without any, the answer reports how many exhibits await review, with no model
  call.
- An unlocated quote matches no message, so its exhibit cannot be confirmed or cited.
- The nonce is a rendering marker recorded in scan metadata, not a secret.

## Modules

| Module | Role |
| --- | --- |
| `_findings/spec.py` | Finding specs, registration, selection of findings for a scan |
| `_findings/scanner.py` | `finding_scanner`, `finding_bundle_scanner`, `add_finding_scanners` |
| `_findings/verify.py` | Quote matching, relocation and excerpts |
| `_findings/derive.py` | The derived view and its per-scan store |
| `_findings/verdicts.py` | The verdict ledger and judge run log |
| `_findings/judge.py` | The exhibit judge |
| `_findings/agreement.py` | Scan and judge agreement lines |
| `_findings/answer.py`, `_findings/synthesis.py` | `answer.md`, synthesis and its checks |
| `_findings/lifecycle.py` | Answer refresh at the end of a scan |
| `_findings/links.py` | Scout View message links |
| `_cli/findings.py` | `scout findings` commands |
| `_view/_api_v2_findings.py` | The Findings tab API |
