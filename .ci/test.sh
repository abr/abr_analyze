#!/usr/bin/env bash
if [[ ! -e .ci/common.sh || ! -e abr_analyze ]]; then
    echo "Run this script from the root directory of this repository"
    exit 1
fi
source .ci/common.sh

# This script runs the test suite and collects coverage information

NAME=$0
COMMAND=$1

if [[ "$COMMAND" == "install" ]]; then
    exe pip install -e ".[tests]"
    exe pip install scipy==1.1.0
elif [[ "$COMMAND" == "script" ]]; then
    exe pytest -v --color=yes --durations 20 --cov=abr_analyze abr_analyze
elif [[ "$COMMAND" == "after_script" ]]; then
    exe eval "bash <(curl -s https://codecov.io/bash)"
elif [[ -z "$COMMAND" ]]; then
    echo "$NAME requires a command like 'install' or 'script'"
else
    echo "$NAME does not define $COMMAND"
fi

exit "$STATUS"
