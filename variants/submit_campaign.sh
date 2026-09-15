#!/bin/bash
# Submit this cluster's share of the protocol campaign (docs/RUNS.md, campaign 2026-09-15).
#
# usage:
#   bash variants/submit_campaign.sh sol          # what Sol should run
#   bash variants/submit_campaign.sh phx          # what Phoenix should run
#   bash variants/submit_campaign.sh phx --dry    # sbatch --test-only: queues nothing, prints
#                                                 # the scheduler's own start estimate per job
#   bash variants/submit_campaign.sh phx vyx      # one economy only
#
# WHY A SCRIPT. Thirteen economies, each with its own memory and walltime, across two clusters
# whose nodes differ by a factor of four in memory. Typing that out is thirteen chances to put a
# 128G job on a 112G node; the allocation lives here instead, with the reason for each row.
#
# HOW THE SPLIT WAS CHOSEN (measured 2026-09-15, not assumed):
#
#   DURABLE -- node memory. Sol public nodes carry ~515 GB and Sol highmem 2.0-2.3 TB; Phoenix
#   public nodes carry 112-125 GB (six at 186 GB) and Phoenix highmem is two nodes at 1.5 TB. So
#   an economy needing more than about 100 GB can only go to Sol. Moving burn-in from 300 to 400
#   raises BGN's and GS21's peaks by 12.5%, because run_oracle.py allocates its arrays over
#   T + burnin: g0235r's measured 77.0 GiB projects to ~87 GiB and g0235s's 74.0 to ~83 GiB.
#   Those two are the only economies that cannot sit comfortably on a Phoenix node, so those two
#   go to Sol and the other eleven go to Phoenix.
#
#   TEMPORARY -- fairshare and backlog. On the day: Sol fairshare 0.0102 with 725 jobs pending on
#   public and 2 idle nodes; Phoenix fairshare 0.1303 -- 12.7x better -- with 12 pending and 253
#   idle. Sol's own scheduler put a public job three days out (Sept 18) and the same job on
#   highmem at six hours (today 17:53), so Sol's share goes to HIGHMEM, not public. Phoenix
#   started every size up to 112G immediately. Re-check before submitting: `sshare -U` and
#   `squeue -h -p public -t PD | wc -l` on each, and --dry, which asks the scheduler directly.
#   If the picture has flipped, move rows between the two lists -- the memory column is the only
#   part that is not negotiable.
#
# SHARED CHECKOUT. Both clusters run from /data/sjpruitt/GitHub/bop-run-upload (realpath of
# ~/GitHub/bop-run-upload on each), so every task on both clusters imports the same Python. NO
# `git pull` ANYWHERE until both queues are empty. Results also land in that one shared
# variants/results; no two economies share a filename, so the two clusters cannot collide there.
# SLURM logs are given a per-cluster prefix below because outslurm/ is shared too.
set -euo pipefail

CLUSTER="${1:?usage: bash variants/submit_campaign.sh <sol|phx> [--dry] [economy]}"
shift || true
DRY=""
ONLY=""
for a in "$@"; do
  case "$a" in
    --dry|--test-only) DRY="--test-only" ;;
    *) ONLY="$a" ;;
  esac
done

# tag | partition | mem | walltime | measured peak at burn-in 300 -> projected at 400
read -r -d '' SOL_JOBS <<'EOF' || true
g0235s|highmem|128G|4-00:00|74.0 GiB -> ~83; longest seed 24.5 h class; calm spells of 240 months
g0235r|highmem|128G|4-00:00|77.0 GiB -> ~87; the binding job of the campaign
EOF

read -r -d '' PHX_JOBS <<'EOF' || true
bgnbase|public|40G|1-00:00|15.8 GiB -> ~18
kpbase|public|48G|1-00:00|29.2 GiB, burn-in already 400
gsbase|public|32G|1-00:00|4.5 GiB -> ~5; GS21 panels are the lightest
vyx|public|48G|2-00:00|30.6 GiB, burn-in already 400
vyg25|public|48G|2-00:00|29.2 GiB; longest KP14 at 7.6 h, x1.75 for the wider grid
g28|public|32G|1-00:00|GS21 class, ~5 GiB
gx7|public|32G|1-00:00|GS21 class; loads five 90-100 MB solutions
bx7|public|32G|1-00:00|GS21 class; loads five 88-100 MB solutions
g0235|public|64G|2-00:00|38.9 GiB -> ~44
g0235f|public|40G|1-00:00|22.0 GiB -> ~25
g0235d|public|40G|1-00:00|15.6 GiB -> ~18; NINE NEW SEEDS, the only economy gaining any
EOF

case "$CLUSTER" in
  sol) JOBS="$SOL_JOBS" ;;
  phx) JOBS="$PHX_JOBS" ;;
  *) echo "unknown cluster '$CLUSTER' (expected sol or phx)" >&2; exit 2 ;;
esac

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
mkdir -p outslurm

n=0
while IFS='|' read -r tag part mem wall why; do
  [ -n "${tag:-}" ] || continue
  [ -z "$ONLY" ] || [ "$ONLY" = "$tag" ] || continue
  n=$((n + 1))
  echo "=== $tag  ($part, $mem, $wall)"
  echo "    $why"
  sbatch ${DRY:+$DRY} \
    -p "$part" --mem="$mem" -t "$wall" --array=0-9 \
    -o "outslurm/${CLUSTER}.seeds.%A.%a.log" \
    --export=ALL,SEED_SPEC="$tag" \
    variants/run_seeds_slurm.sh 2>&1 | sed 's/^/    /'
done <<< "$JOBS"

echo
echo "$n economies, $((n * 10)) tasks, on $CLUSTER."
[ -n "$DRY" ] && echo "DRY RUN: nothing was queued."
exit 0
