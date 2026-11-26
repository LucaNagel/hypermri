#!/usr/bin/env bash
# GUI clients may not inherit the development environment's PATH.
set -euo pipefail

python_path="$(git config --get hypermri.testPython)" || {
    status=$?
    if [[ "$status" != 1 ]]; then
        exit "$status"
    fi
    python_path="$(command -v python)" || python_path=""
}

if [[ "$python_path" != /* || ! -x "$python_path" || -d "$python_path" ]]; then
    echo 'Unit-test Python unavailable. Activate the Python 3.12 development environment and run:' >&2
    echo '  git config --local hypermri.testPython "$(python -c '\''import sys; print(sys.executable)'\'')"' >&2
    echo 'The interpreter must have HyperMRI and its .[dev] dependencies installed.' >&2
    exit 1
fi

script_dir="$(cd -- "${BASH_SOURCE[0]%/*}" && pwd)"
exec "$python_path" "$script_dir/run_tests.py" "$@"
