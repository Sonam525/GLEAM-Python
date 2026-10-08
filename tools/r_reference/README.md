# R reference

gleampy was translated from, and is tested against, one specific state of the
GLEAM R package. This folder holds what is needed to rebuild that state and to
regenerate the golden outputs in `tests/golden/` from it.

## What the reference is

The reference is **not** a released version of the R package. Released
`un-fao/GLEAM` 0.8.0 (branch `main`) lacks chickens, the non-demographic herd
model and `run_emissions_direct()`: 12 of the functions gleampy ports are
missing there, and its results differ. The reference is the union of
unreleased development branches of
[un-fao/GLEAM](https://github.com/un-fao/GLEAM), merged in this order, plus
one adaptation commit:

| Step | Source | Commit | Content |
|---|---|---|---|
| start | `feature/new-species-chk` | `50f3c39dd02fa4969714b1d2657b81b616291769` | chickens (CHK), egg production and allocation |
| merge | `feature/herd-nondemo` | `f3d125475af661389943838a97b2d1836aabd940` | non-demographic herd module, all-herd orchestrator |
| merge | `main` (0.8.0) | `90e416197e89093c4f3a347b263ba805d33d4aac` | released package, CRAN fixes |
| merge | `feature/run-direct-emissions-only` | `aa8915d9c5595ce78fc33db57ed0d34399212bc4` | `run_emissions_direct()` |
| patch | [`patches/0001-*.patch`](patches/) | `eef5adba358059f1a133e58f52519266ef784dd4` | adaptation of `run_emissions_direct()` to the merged herd API |

The merges are clean. The adaptation commit, dated 2026-10-06, changes three
files. `R/run_emissions_direct.R` was written against `main`; it now routes
the herd step through `run_all_herd_module()` (and so gains the
`run_demographic` and `run_nondemographic` arguments), adds the optional
non-demographic herd columns and joins ration quality on
`nondemo_productive_phase_id`, exactly as `run_gleam()` does. Its example herd
table `emissions_direct_input_hrd_data.csv` gains
`prop_nondemo_fem_juv = prop_nondemo_mal_juv = 0`, and its tests point at the
direct-emissions example inputs. The patch is the only content of the
reference that is not in un-fao/GLEAM.

The result is the branch `integration/r-reference` with source tree
`94aef41a7c744fb49a1fcf0f73dc0e9b64d477bc` (`git rev-parse HEAD^{tree}`).
That branch is also published for browsing as
[`integration/r-reference`](https://github.com/Sonam525/GLEAM/tree/integration/r-reference)
in Sonam525/GLEAM
(`git clone -b integration/r-reference https://github.com/Sonam525/GLEAM.git`);
the script below rebuilds it from un-fao/GLEAM and checks it.

The R documentation (`man/`) of that tree documents the chicken,
non-demographic and direct-emissions functions and their input columns;
released 0.8.0's documentation does not. The two arguments the adaptation
commit adds to `run_emissions_direct()`, `run_demographic` and
`run_nondemographic`, are documented only in its roxygen comment in
`R/run_emissions_direct.R`: `man/run_emissions_direct.Rd` was not
regenerated.

## Rebuilding it

Requirements: git and bash (Git Bash on Windows). On Windows, keep the
destination path short (R cannot read files whose full path exceeds 260
characters unless long paths are enabled).

```bash
bash tools/r_reference/build_r_reference.sh ../GLEAM-reference
```

The script clones `https://github.com/un-fao/GLEAM.git` into the new
directory `../GLEAM-reference` (it refuses to touch an existing directory),
recreates the merges above, applies the patch with `git am` and checks that
the resulting tree is the expected one. A second argument replaces the
repository URL, for example a local mirror. The script does not depend on
your git configuration: the merge commits get a neutral identity and are not
signed, and the new clone has `core.autocrlf=false` (so its files are
byte-identical on every platform) and `core.longpaths=true`.

Commit hashes differ from the original branch (`9e21be0` after the merges,
`eef5adb` after the patch), because the merges and the patch are committed
again locally; the script compares tree hashes. If un-fao/GLEAM no longer
serves one of the commits above (for example after a branch is deleted),
pass `https://github.com/Sonam525/GLEAM.git` as the second argument: its
`integration/r-reference` branch contains all of them, and the script still
checks the resulting tree.

## Running the R package

Install R (>= 4.4, as required by the package) and the packages it needs:

```r
install.packages(c("data.table", "cli", "pkgload", "testthat"))
```

The golden outputs were generated with R 4.6.1, data.table 1.18.6.1, cli
3.6.6 and pkgload 1.5.3. Load the reference without installing it:

```r
suppressMessages(pkgload::load_all("../GLEAM-reference", quiet = TRUE, export_all = TRUE))
```

Put longer R code in a script file and run it with `Rscript file.R`; long
`Rscript -e` one-liners are fragile, especially with shell quoting on
Windows. R evaluates the model row by row, so keep inputs small when
exploring.

The reference passes its own test suite: `testthat::test_dir()` on
`tests/testthat` (with the package loaded as above) reports 306 tests and 733
expectations (122 of them in `test-run_emissions_direct.R`), with no failures
or errors. The direct-emissions tests emit 4 expected warnings ("no
non-demographic rows were found").

## Regenerating golden outputs safely

`generate_golden.R` runs every R example used by the parity tests and writes
one CSV per output table, with doubles at 17 significant digits:

```bash
Rscript tools/r_reference/generate_golden.R <R source> <out_dir> [case_regex]
```

Read this before running it:

* **Never write straight into `tests/golden/`.** `out_dir` is required, and
  the script refuses to write into `tests/golden`. For every case it runs, it
  first deletes that case's files in `out_dir`, and when R fails it writes
  `ERROR.txt` in their place. Always pass a new, empty directory:

  ```bash
  Rscript tools/r_reference/generate_golden.R ../GLEAM-reference ../golden-new
  ```

* **Compare before you copy.** Check what changed, then copy only the cases
  you meant to change:

  ```bash
  git diff --no-index --stat tests/golden ../golden-new
  ```

* **One case at a time.** The third argument is a regular expression matched
  against case names (`grepl`); anchor it to select a single case. It still
  writes into `out_dir`, so keep that a scratch directory:

  ```bash
  Rscript tools/r_reference/generate_golden.R ../GLEAM-reference ../golden-new '^weights_module$'
  git diff --no-index tests/golden/weights_module ../golden-new/weights_module
  ```

* Some cases read inputs from `tests/data/` (written by
  `tests/data/build_golden_inputs.py`), so run the script from a full
  checkout of this repository. A few cases are expected to fail in R; their
  golden is the R error message in `ERROR.txt`, and the tests check that
  Python raises the same error.
* The script needs the complete reference. On another R tree, cases fail or
  give different tables, and the script can stop part-way (it reads the
  direct-emissions example inputs whatever `case_regex` selects).

## Randomised parity scenarios

`tools/parity/` perturbs the pipeline examples, runs each scenario through R
and Python and compares all outputs. Keep its output directory outside the
repository (or in the ignored `scenarios/` folder):

```bash
python tools/parity/make_scenarios.py ../parity-scenarios 10 20261006
Rscript tools/parity/run_scenarios.R ../GLEAM-reference ../parity-scenarios
python tools/parity/compare_scenarios.py ../parity-scenarios
```
