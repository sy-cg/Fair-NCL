# Revision Delivery And Public Release Checklist

## Local Release Identity

The prepared revision is identified by
`docs/release/revision_submission_release.json`. It records:

- `schema_version` and `release_id`;
- `code_snapshot.archive`, `content_sha256`, `archive_sha256`, and `file_count`;
- `source_data.archive`, `archive_sha256`, and `file_count`;
- `manuscript.pdf` and `sha256`;
- `public_release.status`, `repository_url`, and `git_commit`;
- `historical_run_commits`, which remains `not_recorded` for the archive.

The local release content hash identifies the exact prepared code payload. It
is not a Git commit and is not evidence of the code version that produced
each historical run. All 595 archived environment records retain their missing
historical commit fields.

## Delivery Layout

The journal delivery is assembled under `submission_package/`:

```text
submission_package/
  README.md
  SHA256SUMS
  revised_locations.csv
  code/
    fair_ncl_revision_code.zip
  data/
    source_data.zip
  manuscript/
    clean.pdf
    latex_source.zip
  response/
    (Fair-NCL response documents)
```

The response and revised-location table refer to the clean manuscript. They
do not assert yellow highlighting. A separately titled paper's response is
excluded from this delivery.

The public-release candidate consists of the code and aggregate-data payloads.
The journal response and review correspondence are not part of that public
candidate. Preparing the journal delivery does not publish its contents.

## Verify The Delivered Files

Run from `submission_package/` after obtaining the delivery files:

```bash
sha256sum -c SHA256SUMS
```

The checksum list verifies the listed delivered files. The release manifest
provides the separate content and archive identities for the code and source
data. Neither a checksum nor an archive filename establishes that a public
repository already hosts the revision.

Check the final generated inventory against these expected data properties:

| Family | Expected contents |
|---|---|
| S1 primary | 96 overall-utility and marginal absolute-gap comparisons |
| S1 complete | 672 Fair-NCL comparisons; the primary 96 are an exact subset, with existing correction families preserved |
| S2 | 66 points: nine methods in each SASRec block, eight in each remaining block; Fair-NCL on four of eight Pareto fronts |
| S3--S5 | 320 archived core-comparison run costs, validation/search audit, and eight Fair-NCL/backbone paired-cost settings |
| S6--S7 | Component controls and the separately documented CoSeRec results and selection |
| S8 | 220 run records, 44 summaries, 504 paired contrasts |
| S9 | 18 three-seed points from 42 unique source jobs; centers reused across axes |
| S10 | Five settings, 50 method-run records, ten summaries, 70 contrasts; minimum Holm-adjusted p=0.3125; ten new SASRec runs included |

Inspect the actual manifest and document-validation records rather than
assuming that every check has passed because this list exists. The analysis
commands in [REPRODUCIBILITY.md](REPRODUCIBILITY.md) explain how the files are
generated from completed runs. Assembly does not launch new training, establish
historical software identity, or assert that the full training archive was
rerun in a fresh environment.

## Check Exclusions

The prepared public code/data candidate must exclude:

- third-party raw datasets and downloaded data archives;
- processed user-level sequences and per-user prediction outputs;
- checkpoints, model/embedding caches, and large training artifacts;
- credentials, passwords, API tokens, local account/environment configuration;
- `.codex/`, IDE state, and Python bytecode;
- journal review correspondence and another manuscript's response files.

Reused raw data remain available through their original hosts, with filenames,
checksums, and deterministic preprocessing documented. Shared source data are
aggregate estimates, configurations, coordinates, and environment records.

## Public Publication Remains Separate

Before an external upload, the manifest uses
`public_release.status = not_published_pending_author_approval` and a null
public revision commit. There is no invented DOI or placeholder presented as
a real identifier. The local package can be reviewed in this state.

After an authorized upload or release has completed:

1. Record its actual public URL and commit, plus any issued archive DOI or
   accession; retain the local content hash for comparison.
2. Download the released payload and verify it against the recorded checksums.
3. Update the public status in the release manifest and the manuscript's data
   statement to match what is actually accessible.
4. Continue to identify historical training commits as unrecorded; publishing
   the current revision cannot recover them.
