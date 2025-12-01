#!/bin/bash
# Monitor when jobs start running

echo "Monitoring jobs at $(date)"
echo "================================"

while true; do
    RUNNING=$(squeue -u eliu354 -t R | wc -l)
    if [ $RUNNING -gt 1 ]; then
        echo "✓ Jobs started running at $(date)!"
        squeue -u eliu354 -t R -o "%.10i %.20j %N"
        break
    fi
    echo "$(date): Still waiting... (Running: $((RUNNING-1)))"
    sleep 120  # Check every 2 minutes
done
