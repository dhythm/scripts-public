#!/usr/bin/env bash
OUTDIR="$HOME/Screenshots/region"
INTERVAL=5
X=200; Y=120; W=800; H=600

mkdir -p "$OUTDIR"
echo "Capturing rect $X,$Y,$W,$H every $INTERVAL sec. Ctrl+C to stop."

while true; do
  TS=$(date +"%Y%m%d_%H%M%S")
  screencapture -x -R"${X},${Y},${W},${H}" "$OUTDIR/${TS}.png"
  echo "$OUTDIR/${TS}.png"
  sleep "$INTERVAL"
done
