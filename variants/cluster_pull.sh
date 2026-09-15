#!/bin/bash
# Pull the shared cluster checkout of bop-run-upload.
#
# Jobs write their outputs into this checkout (variants/results/, experiments/registry/). Those files
# are copied to the laptop, committed and pushed there, and then the cluster pulls. At that point the
# cluster still holds its own UNTRACKED copies of files the incoming commits bring in as TRACKED, and
# a plain `git pull` refuses to overwrite them -- even byte-identical ones. This script checks that
# EVERY file the pull would overwrite (untracked, or tracked and modified) is byte-identical to what
# origin carries, and only then clears those copies and fast-forwards. A single difference aborts
# with the list, touching nothing. So the pull can never discard cluster output the laptop did not
# commit exactly.
#
# Run on the cluster, inside the checkout, with no job that imports from it running on either cluster:
#     cd /data/sjpruitt/GitHub/bop-run-upload && bash variants/cluster_pull.sh
#
# Exit codes: 0 pulled (or already up to date); 2 not a checkout or fetch failed; 3 a blocking file
# differs from origin (nothing touched); 4 the checkout cannot fast-forward (nothing touched).
#
# This file lives in the tree it updates, so a pull can rewrite it while it runs, and bash reads a
# script as it executes rather than all at once. Everything is therefore inside `main`, and the last
# line calls it and exits on the same line: bash has parsed the whole file before the pull can change
# it. A change to this script takes effect from the pull AFTER the one that brings it in.
# CLUSTER_PULL_BRANCH overrides the branch (default main).

main() {
  set -u
  local top branch up nm nu bad f
  top=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "ABORT: not inside a git checkout" >&2; return 2; }
  cd "$top" || return 2
  branch=${CLUSTER_PULL_BRANCH:-main}
  up="origin/$branch"

  # Private scratch space, removed on exit: no fixed names under /tmp that a concurrent run, or a
  # run on the other cluster's login node, could collide with.
  CLUSTER_PULL_TMP=$(mktemp -d "${TMPDIR:-/tmp}/cluster_pull.XXXXXX") || { echo "ABORT: mktemp failed" >&2; return 2; }
  trap 'rm -rf "$CLUSTER_PULL_TMP"' EXIT
  local t=$CLUSTER_PULL_TMP

  git fetch -q origin "$branch" || { echo "ABORT: git fetch origin $branch failed" >&2; return 2; }
  if ! git merge-base --is-ancestor HEAD "$up"; then
    echo "ABORT: HEAD $(git rev-parse --short HEAD) is not an ancestor of $up, so the checkout cannot fast-forward; nothing touched" >&2
    return 4
  fi

  git diff --name-only HEAD "$up" | sort > "$t/incoming"
  git diff --name-only HEAD       | sort > "$t/modified"
  git ls-files --others --exclude-standard | sort > "$t/untracked"
  comm -12 "$t/incoming" "$t/modified"  > "$t/modblock"
  comm -12 "$t/incoming" "$t/untracked" > "$t/untblock"

  # A blocking file is safe to clear only if its bytes are exactly the blob origin carries at that
  # path. A path origin deletes has no blob, so it fails the check and aborts.
  same_as_origin() { [ "$(git hash-object -- "$1")" = "$(git rev-parse -q --verify "$up:$1" 2>/dev/null)" ]; }
  bad=0
  while IFS= read -r f; do same_as_origin "$f" || { echo "  DIFFERS (modified):  $f" >&2; bad=1; }; done < "$t/modblock"
  while IFS= read -r f; do same_as_origin "$f" || { echo "  DIFFERS (untracked): $f" >&2; bad=1; }; done < "$t/untblock"
  if [ "$bad" = 1 ]; then
    echo "ABORT: the files above are not what $up carries; nothing touched" >&2
    return 3
  fi

  nm=$(wc -l < "$t/modblock" | tr -d ' '); nu=$(wc -l < "$t/untblock" | tr -d ' ')
  while IFS= read -r f; do git checkout -q -- "$f"; done < "$t/modblock"
  while IFS= read -r f; do rm -f -- "$f"; done < "$t/untblock"
  echo "cleared $nm modified + $nu untracked, all byte-identical to $up"

  git merge -q --ff-only "$up" || { echo "ABORT: fast-forward to $up failed after clearing" >&2; return 4; }
  echo "HEAD: $(git log --oneline -1)"
  echo "dirty: $(git status --porcelain | grep -vc '^??') modified, $(git status --porcelain | grep -c '^??') untracked"
  return 0
}

main "$@"; exit $?
