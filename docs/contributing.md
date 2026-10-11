# Contributing

Contributions, bug reports, and feature requests are welcome. Open an issue or submit a PR on [GitHub](https://github.com/marwinsteiner/pysvi).

## Development setup

```bash
git clone https://github.com/marwinsteiner/pysvi.git
cd pysvi
uv sync --dev
uv run pytest
```

## Adding a parametrization

New models subclass `Parametrization` in `src/pysvi/models.py` and implement `calibrate(k, w_target, **kwargs)` and `total_variance(k, params)`. Register the model in the `get_model` factory in `src/pysvi/calibration.py`, export it from `__init__.py`, add tests mirroring the existing per-model suites, and add a docs page under `docs/models/` following the shared structure (overview, model, parameters, usage, arbitrage behaviour, references).

Any change to the public API must update the {doc}`examples <examples>` and the documentation **in the same pull request** — the examples exercise every public endpoint and parameter, and drift between them and the library is treated as a bug.

## How releases work: the full lifecycle

`svi-py` ships on a fully automated weekly release train. Nothing is released by hand; understanding the moving parts helps you land a change in a particular version.

### The lifecycle at a glance

```text
issue --> milestone (vX.Y.Z, due date = release Sunday)
      --> feature branch + PR (title "vX.Y.Z: theme", body = release notes)
      --> review, CI green
      --> maintainer adds the `release-ready` label
      --> Sunday 16:00 UTC: release train
            merges the PR into main
            creates annotated tag vX.Y.Z (message = PR title + body)
            dispatches the CI/CD pipeline at the tag
            closes the milestone's issues and the milestone
      --> CI/CD at the tag
            runs the test suite (numba extra, coverage to Codecov)
            creates the GitHub Release (title = bare tag, notes = tag message)
            builds and publishes to PyPI (trusted publishing)
```

### Versioning

Versions come from git tags via `hatch-vcs` — there is no version string in the source. Pushing a tag `vX.Y.Z` **is** the deployment trigger; the package version on PyPI, the GitHub Release, and the tag are one and the same object. Between releases, local builds carry a `.devN` suffix derived from the distance to the last tag.

### Milestones are the release calendar

Each planned version is an open GitHub milestone whose **title is the tag** (`v1.0.0`) and whose **due date is the Sunday it ships**. Issues are attached to the milestone they ship in. Rescheduling a release means moving the milestone's due date — nothing else. The train only ever considers the *earliest overdue* open milestone, so if a week is missed the backlog catches up one version per Sunday, in order.

### Pull request conventions

- One PR per version. The **PR title becomes the tag's subject line** — write it as `vX.Y.Z: short theme`. The **PR body becomes the release notes** verbatim, so write it for users, not reviewers.
- Attach the milestone for the target version.
- PRs can be built and reviewed at any time; they sit unmerged until their scheduled Sunday. Work for later versions is **stacked**: each feature branch is cut from the previous version's branch, so every PR's diff shows only its own changes and narrows automatically as earlier PRs merge. A fix that affects several stacked PRs is made on the lowest affected branch and forward-merged upward.
- The `release-ready` label is the human gate: the train never merges a PR without it. Adding the label is the maintainer's sign-off that the PR may ship on its milestone's date.

### The release train (`.github/workflows/release-train.yml`)

Runs every Sunday at 16:00 UTC (and on manual `workflow_dispatch` for off-schedule releases). Each run releases **at most one version**:

1. Find the earliest open milestone whose due date has passed. None due: no-op.
2. Find an open PR carrying both that milestone and the `release-ready` label. None: no-op (the release waits, and is caught up on a later run once labelled).
3. Verify the PR's checks are green — failing or pending checks abort the run with an error rather than shipping.
4. Merge the PR into `main` (merge commit).
5. Create an **annotated tag** named after the milestone, with the PR title and body as the tag message, and push it.
6. Dispatch the CI/CD pipeline at the tag explicitly (tags pushed with the workflow's own token do not fire `on: push` workflows — a GitHub Actions recursion guard — but `workflow_dispatch` is exempt).
7. Close every remaining open issue on the milestone (with a comment naming the release and PR), then close the milestone itself.

### The CI/CD pipeline (`.github/workflows/python-publish.yml`)

Runs on every push and PR to `main` (tests only) and at every `v*` tag (full deployment):

- **test** — full suite with the numba extra installed, coverage uploaded to Codecov.
- **release** (tags only) — creates the GitHub Release. The notes are read from the annotated tag's message (the pipeline re-fetches the real tag ref first, because checkout peels annotated tags on dispatched runs); the release title is the bare tag name.
- **deploy** (tags only, after test + release) — builds sdist and wheel and publishes to PyPI via trusted publishing (OIDC; no long-lived credentials).

Documentation on [Read the Docs](https://pysvi.readthedocs.io) is built from `main`, so the site updates as release PRs merge.

### What this means for a contribution

Open an issue, agree on which milestone it belongs to, and target your PR accordingly. If it lands in the current release PR's scope, it may be merged into that PR's branch; otherwise it waits for its own version. You never need to touch tags, versions, changelogs, or PyPI — writing a good PR description *is* writing the release notes.

## Wanted: the original Gamma-Vanna-Volga paper

The Gamma-Vanna-Volga parametrization is something of a holy grail in the quant vol surface literature and would be a great addition to this library. If you have a copy of the original paper, please send it to [marwin.steiner@gmail.com](mailto:marwin.steiner@gmail.com).

## License

MIT
