#!/bin/bash
set -euo pipefail
cd /submission
exec python3 -u /submission/agent.py "$@"
