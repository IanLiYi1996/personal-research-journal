#!/usr/bin/env bash
# Run one scheduled digest task headlessly. Invoked by systemd user timers
# (~/.config/systemd/user/digest@.timer). Usage: scheduled_run.sh <name>
# where ops/schedule/<name>.prompt holds the prompt.
# Tasks are serialized with a shared lock so two runs never commit at once.
set -euo pipefail
name="$1"
repo="$HOME/research/personal-research-journal"
export PATH="$HOME/.local/bin:$HOME/.nvm/versions/node/v22.22.0/bin:$PATH"
logdir="$HOME/.local/state/digest-runs"; mkdir -p "$logdir"
log="$logdir/$(date -u +%Y%m%dT%H%M%SZ)-$name.log"
cd "$repo"
prompt="$(cat "ops/schedule/$name.prompt")"
exec 9>"$logdir/.lock"
echo "[$(date -u +%FT%TZ)] waiting for lock" >>"$log"
flock 9
echo "[$(date -u +%FT%TZ)] start $name" >>"$log"
timeout 3h claude -p --permission-mode bypassPermissions "$prompt" >>"$log" 2>&1 || echo "exit $?" >>"$log"
echo "[$(date -u +%FT%TZ)] end $name" >>"$log"
