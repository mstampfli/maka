#!/usr/bin/env bash
# Every `neg_*.maka` must fail to compile.
set -u
cd "$(dirname "$0")/.."
cargo build --quiet || exit 1
MAKAC=./target/debug/makac
fail=0
for src in tests/programs/neg_*.maka; do
    name=$(basename "$src" .maka)
    out=$("$MAKAC" "$src" --emit-c -o /tmp/__neg 2>&1)
    rc=$?
    if [ "$rc" = "0" ]; then
        echo "FAIL $name: compiled but should have failed"
        echo "$out" | sed 's/^/    /'
        fail=1
    else
        echo "ok   $name (rejected: $(echo "$out" | head -1))"
    fi
done
exit "$fail"
