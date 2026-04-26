# Sync Policy (tasai)

This repository is the **code source of truth**.

## Relationship to tasai_paper_clean

`tasai_paper_clean` should either:

1. install a pinned git revision of this repository into its environment, or
2. vendor a minimal snapshot exported from a tagged commit in this repository.

### Recommended workflow

1. Develop and validate changes here (`tasai`).
2. Commit the paper-ready state here.
3. Create an annotated tag for the manuscript freeze.
4. Export a minimal snapshot with `scripts/export_paper_snapshot.sh` if the
   paper bundle must be self-contained.
5. Record the exact tag and commit hash in the paper repo reproducibility notes.

Do not maintain long-term divergence between this repo and vendored copies.
Do not vendor duplicate top-level module trees, virtual environments, caches,
or build artifacts into the paper bundle.
