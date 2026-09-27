#!/bin/bash
set -e
source /home/vlad/Explorer/bin/env.sh
exec python3 /home/vlad/Explorer/bin/commission-arm.py "$@"
