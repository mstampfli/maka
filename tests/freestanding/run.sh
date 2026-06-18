#!/usr/bin/env bash
# Verifies the --freestanding flag: Maka source → no-libc C → static ELF.
set -eu
cd "$(dirname "$0")/../.."
cargo build --release --quiet
MAKAC=./target/release/makac
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

"$MAKAC" --freestanding tests/freestanding/kernel_hello.maka --emit-c -o "$tmp/k.c" >/dev/null

# Confirm the generated C has NO libc dependencies.
if grep -E '^#include <stdio|<stdlib|<pthread|<string|<sys/' "$tmp/k.c" >/dev/null; then
    echo "FAIL: freestanding output references libc"
    grep -E '^#include <stdio|<stdlib|<pthread|<string|<sys/' "$tmp/k.c"
    exit 1
fi

# Compile + link with -ffreestanding + -nostdlib + stub runtime.
gcc -ffreestanding -nostdlib -fno-stack-protector -nostartfiles \
    -Wno-unused-function \
    -c "$tmp/k.c" -o "$tmp/k.o"
gcc -ffreestanding -nostdlib -fno-stack-protector -nostartfiles \
    -c tests/freestanding/stub_runtime.c -o "$tmp/stub.o"
gcc -nostdlib -static -o "$tmp/img" "$tmp/k.o" "$tmp/stub.o"
file "$tmp/img" | grep -q "ELF .* executable" || { echo "FAIL: link did not produce an ELF"; exit 1; }

echo "ok   freestanding kernel build"
