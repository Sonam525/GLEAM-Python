#!/usr/bin/env bash
# Rebuild the R reference that gleampy was ported from and is tested against.
#
# Usage (from anywhere):
#   bash tools/r_reference/build_r_reference.sh [DEST_DIR] [SOURCE_REPO]
#
#   DEST_DIR     directory for the new R checkout; must not exist yet
#                (default: ../GLEAM-reference, next to this repository)
#   SOURCE_REPO  URL or local path of the GLEAM R repository
#                (default: https://github.com/un-fao/GLEAM.git)
#
# The reference is not a released version of the R package. It is the union
# of four upstream commits of un-fao/GLEAM, merged in this order on top of the
# first one, plus one local adaptation commit that is shipped as a patch in
# tools/r_reference/patches/:
#
#   feature/new-species-chk            50f3c39  (start point)
#   feature/herd-nondemo               f3d1254  (merged)
#   main (0.8.0)                       90e4161  (merged)
#   feature/run-direct-emissions-only  aa8915d  (merged)
#   patches/0001-*.patch               eef5adb  (applied with git am)
#
# The script clones SOURCE_REPO into DEST_DIR, recreates the branch
# integration/r-reference there and checks that its source tree is
# byte-identical to the one the golden outputs in tests/golden were generated
# from. It never modifies an existing directory. It needs git and bash only;
# running the R package afterwards needs R (>= 4.4) with data.table, cli and
# pkgload (testthat for the R test suite). See tools/r_reference/README.md.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEST="${1:-$REPO_ROOT/../GLEAM-reference}"
SOURCE="${2:-https://github.com/un-fao/GLEAM.git}"
PATCH_DIR="$SCRIPT_DIR/patches"
BRANCH="integration/r-reference"

# Upstream commits (full hashes) in merge order.
START_COMMIT="50f3c39dd02fa4969714b1d2657b81b616291769"   # feature/new-species-chk
MERGES=(
  "f3d125475af661389943838a97b2d1836aabd940 feature/herd-nondemo"
  "90e416197e89093c4f3a347b263ba805d33d4aac main"
  "aa8915d9c5595ce78fc33db57ed0d34399212bc4 feature/run-direct-emissions-only"
)
# Source trees of the original integration branch: after the three merges
# (commit 9e21be0) and after the adaptation commit (eef5adb).
EXPECTED_MERGED_TREE="329b6395b34c571a094818ec8d9ab4f787e777f2"
EXPECTED_FINAL_TREE="94aef41a7c744fb49a1fcf0f73dc0e9b64d477bc"

die() { echo "build_r_reference.sh: error: $*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git is not installed"
[ -e "$DEST" ] && die "'$DEST' already exists; pass a new directory as the first argument"
shopt -s nullglob
PATCHES=("$PATCH_DIR"/*.patch)
shopt -u nullglob
[ "${#PATCHES[@]}" -gt 0 ] || die "no patch found in $PATCH_DIR"

# Merges and git am create commits: use a fixed, neutral identity and no
# signing (these local commits are not kept; only their trees are compared),
# so the build does not depend on (or write) the user's git configuration.
# Line-ending conversion is switched off so the trees can be compared byte for
# byte on every platform.
git_ref() {
  git -C "$DEST" \
    -c user.name="gleampy reference build" \
    -c user.email="reference-build@localhost" \
    -c core.autocrlf=false \
    -c commit.gpgsign=false \
    -c advice.detachedHead=false \
    "$@"
}

ensure_commit() {
  local sha="$1"
  if ! git -C "$DEST" cat-file -e "${sha}^{commit}" 2>/dev/null; then
    echo "Fetching $sha ..."
    git -C "$DEST" fetch --quiet origin "$sha" \
      || die "commit $sha is not available from '$SOURCE'"
  fi
}

echo "Cloning $SOURCE into $DEST ..."
# --origin: ensure_commit fetches from "origin", whatever clone.defaultRemoteName
# says. core.longpaths lets Git for Windows check out the deepest example file
# under a long DEST (it is ignored elsewhere).
git clone --quiet --no-checkout --origin origin \
  --config core.longpaths=true --config core.autocrlf=false "$SOURCE" "$DEST"

ensure_commit "$START_COMMIT"
git_ref checkout --quiet --force -B "$BRANCH" "$START_COMMIT"

for entry in "${MERGES[@]}"; do
  sha="${entry%% *}"
  name="${entry#* }"
  ensure_commit "$sha"
  echo "Merging $name (${sha:0:7}) ..."
  git_ref merge --quiet --no-ff --no-edit \
    -m "Merge $name (${sha:0:7}) into $BRANCH" "$sha" \
    || die "merge of $name failed (see git's message above)"
done

tree="$(git -C "$DEST" rev-parse 'HEAD^{tree}')"
[ "$tree" = "$EXPECTED_MERGED_TREE" ] \
  || die "tree after the merges is $tree, expected $EXPECTED_MERGED_TREE"

for patch in "${PATCHES[@]}"; do
  echo "Applying $(basename "$patch") ..."
  git_ref am --quiet "$patch" || die "could not apply $patch"
done

tree="$(git -C "$DEST" rev-parse 'HEAD^{tree}')"
[ "$tree" = "$EXPECTED_FINAL_TREE" ] \
  || die "final tree is $tree, expected $EXPECTED_FINAL_TREE"

cat <<EOF

R reference rebuilt in $DEST
  branch $BRANCH, tree $tree (identical to the reference used for tests/golden)

Next steps (from the gleampy repository root):
  Rscript tools/r_reference/generate_golden.R "$DEST" <new-empty-directory>
See tools/r_reference/README.md before writing anything into tests/golden.
EOF
