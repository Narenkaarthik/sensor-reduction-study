# Commit map for the 2026-10-31 history rewrite

On 2026-10-31, before this repository was made public, the author, committer and
tagger e-mail addresses in every commit and tag were replaced with the author's
GitHub `users.noreply.github.com` address. Rewriting an identity changes the
commit hash, so **every commit hash in this repository changed on that date.**

Nothing else changed. For each of the 13 commits the rewrite is verified to
preserve:

- the **tree** — the exact file contents, byte for byte;
- the **commit message**;
- the **author date and the committer date**.

So the code, the data, the results and every number are untouched. Only the
e-mail field differs.

## Why this file exists

`PROTOCOL.md` §12.1 and the `run_metadata.json` written beside every output
record the commit that produced each result. Those records contain **pre-rewrite
hashes**, and they were deliberately left as they are:

- the `run_metadata.json` files are written by the pipeline, not by hand, and the
  repository's rule is that no generated file is ever hand-edited. Editing one to
  name a different commit would make it claim something the code never observed.
- `PROTOCOL.md` §12.1(a) **quotes real log output**. Rewriting hashes inside a
  quoted program output would falsify it.

This table is therefore the translation layer. To follow a provenance reference
recorded anywhere in this repository, look the old hash up here and use the new
one.

### Example

`PROTOCOL.md` §12.1(a) reports that the feature table was extracted at
`af5b0605` and the experiments were run at `1f82015d`, and invites the reader to
confirm that the intervening code change was confined to docstrings. In the
public history that check is:

```bash
git diff e5aa6730 0e0ee988 -- 03_run_experiments.py
```

## Commits

Listed oldest first. Both columns are full 40-character hashes.

| # | Pre-rewrite | Post-rewrite | Subject |
|---|---|---|---|
| 1 | `632bc54fd4537bd3b415526f4ad3724e99bd7d93` | `9f43ad1ddc7cce2f557022880b55ba363c3b1bc9` | sensor-reduction study: pipeline , tests, protocol and paper skeleton |
| 2 | `cc871fd760eda7872236d11c7032028aa7911c68` | `dde9d9351525179f4a9b9a4bd319eb42c66596d6` | PROTOCOL: record the history squash and authorship correction |
| 3 | `f0e531d6cd05c1c45b9023d106a8a32e36fec88b` | `a2e611de05aba6839777a6c0f20ad9408e91305c` | speed_hz from the file name; keep the tachometer estimate alongside |
| 4 | `af5b0605e7aa164e1f78237a0aa44333d0866bb4` | `e5aa67303deab121a631299e00ebd75651b3e425` | Exclude the tachometer: 7 sensors, 127 subsets |
| 5 | `ade6d23e94a48f35735d8de033fd7a4b118c398e` | `c4955c893b9ecffae6ff2080a7e9bf7dc30dff8a` | Level B: assert channel counts against ALL_CHANNELS, not SENSORS |
| 6 | `1f82015dc875b6a27a9315e9b1aedb1d3abc26f6` | `0e0ee9885713541993b963d080c8fe869d1bf321` | Split mode: speed primary, recording secondary; fix stale counts |
| 7 | `edb0cc6d78eda545682283abc34f916aae0e5db7` | `52c30edde699f38640a2a460ca2ff7a12a2bbfca` | Per-mode analysis outputs, fig5 k=1 variant, Holm post-hoc, results tracked |
| 8 | `b566cd7c5c389741a18db515e9980459cc25647a` | `d743e80566cb364deaa4a89e4f4d6ca55415d61c` | Secondary 8-sensor analysis: results, tables and figures |
| 9 | `24a1a0f51cda6637757e3193fb09bab25a6c8a30` | `9f25f60024e4d19bf3e16862f5f20d3be086d5b7` | Derive the candidates header; record the secondary outcome and learnability |
| 10 | `24f82cd3df377fe8d9b469612f36745224113599` | `6cb2f88922628fea61c6b994d62230a9fc824f6f` | Decisive artefact test; scope the dirty check to source files |
| 11 | `756e5afec93f2b806161d6ed60e06bb01bb6b990` | `baa11e572bc7424d791ba65af8bd180872d57a0b` | Test: assert the new dirty-warning wording |
| 12 | `cffc129b3b0b09cd1b88ff48094027d1582a6a71` | `f98ac946851ee20cb996ef769afb239302320329` | Symmetrical supplementary tables; results inventory; record the correction |
| 13 | `8190bbb2ec707d63b89d1bd9afb2f5acc16032c2` | `91276c0e5cc8ea42f72f798a9093b2e914be251f` | Release packaging: README front door, CI, licence and citation |

## Tags

Both annotated tags were recreated on the rewritten commits. The tag messages
are byte-identical to the originals and the tagger dates are unchanged; only the
tagger e-mail differs.

| Tag | Points at (pre-rewrite) | Points at (post-rewrite) |
|---|---|---|
| `v1.0.1-protocol` | `632bc54fd4537bd3b415526f4ad3724e99bd7d93` | `9f43ad1ddc7cce2f557022880b55ba363c3b1bc9` |
| `v1.1-protocol` | `1f82015dc875b6a27a9315e9b1aedb1d3abc26f6` | `0e0ee9885713541993b963d080c8fe869d1bf321` |

## Verifying this map yourself

The claim that only the e-mail changed is checkable without trusting this file.
For any row, the two commits must have the same tree, message and dates:

```bash
git show -s --format='%T %aI %cI %s' <post-rewrite-hash>
```

Compare that against the same command run on a pre-rewrite clone, if you have
one. The pre-rewrite history is not published; it was preserved locally as a
`git bundle` at the time of the rewrite.
