# Releasing: going public and minting the archival DOI

Written 2026-11-09. Both steps below need your GitHub and Zenodo accounts, so
they cannot be scripted from inside this repository.

**The order matters.** Zenodo only archives releases created *after* the
repository is switched on in Zenodo. Create the release first and nothing is
archived, and you have to cut a second release to trigger it.

---

## Before you start: the pre-publication checks

All green as of commit `c38cc73`:

| Check | State |
|---|---|
| Windows username in tracked files | **0** occurrences |
| API keys, tokens, private keys | none |
| Absolute machine paths in metadata | none; `run_metadata.json` records repo-relative paths |
| Git history e-mail | rewritten to the GitHub noreply address 2026-10-31, see [COMMIT_MAP.md](COMMIT_MAP.md) |
| Dataset | not committed; `data/` and `features/` are gitignored |
| Tests | 139 passed, 0 skipped |

One deliberate exception: `paper/main.tex` carries
`s.narenkaarthik@gmail.com` as the corresponding-author address. That is
normal for a paper, but it does become public. Swap it for an institutional
address first if you would rather it did not.

---

## Step 1 — make the repository public

GitHub web UI, since the `gh` CLI is not installed here:

1. <https://github.com/Narenkaarthik/sensor-reduction-study> → **Settings**
2. Scroll to **Danger Zone** → **Change repository visibility** → **Make
   public**
3. Type the repository name to confirm

**This is irreversible in practice.** Making it private again does not undo
forks, caches or archive crawlers.

---

## Step 2 — connect Zenodo, then release

1. Sign in at <https://zenodo.org> with **Log in with GitHub** and authorise
   it.
2. Go to <https://zenodo.org/account/settings/github/>. Your repositories are
   listed; a private repository will not appear, which is why Step 1 comes
   first.
3. Flip the toggle **on** for `sensor-reduction-study`.
4. Back on GitHub: **Releases** → **Draft a new release**.
   - Tag: `v1.0.0` (the existing `v1.0.1-protocol` and `v1.1-protocol` tags
     mark protocol freezes, not the release)
   - Title: the paper's title
   - Description: a line saying this is the code and stored predictions
     accompanying the paper
   - **Publish release**
5. Zenodo picks it up within a few minutes and mints two DOIs:
   - a **concept DOI** that always resolves to the newest version
   - a **version DOI** for this exact release

**Cite the concept DOI in the paper.** It keeps resolving if you later cut a
`v1.0.1`.

---

## Step 3 — put the DOI in three places

Once Zenodo shows the DOI (format `10.5281/zenodo.NNNNNNN`):

**1. `paper/main.tex`** — the Conclusion's availability sentence. Find:

```latex
\url{https://github.com/Narenkaarthik/sensor-reduction-study}, with
```

and make it:

```latex
\url{https://github.com/Narenkaarthik/sensor-reduction-study} and archived at
\url{https://doi.org/10.5281/zenodo.NNNNNNN}, with
```

**2. `CITATION.cff`** — uncomment the `identifiers` block and fill the value.

**3. `README.md`** — a DOI badge, if you want one:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.NNNNNNN.svg)](https://doi.org/10.5281/zenodo.NNNNNNN)
```

Then recompile the paper: `pdflatex`, `bibtex`, `pdflatex`, `pdflatex`.

---

## What is still open after this

- `CITATION.cff` has no `preferred-citation`. That needs the real conference
  or journal details and cannot be filled until the paper is accepted.
- The Wilcoxon DOI in `paper/refs.bib` rests on a single source. Confirm it
  against Crossref or drop the field; the volume, number, pages and year are
  independently confirmed and are enough on their own.
