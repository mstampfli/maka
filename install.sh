#!/usr/bin/env bash
# Install the Maka toolchain (`maka` project tool + `makac` compiler) onto PATH.
#
#   ./install.sh                 # build release, install to the default bindir
#   BINDIR=~/.local/bin ./install.sh
#   ./install.sh --no-build      # install the existing target/release binaries
#
# `maka` locates `makac` as a sibling, so both land in the same directory.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"

# Default bindir: ~/.cargo/bin (present for any Rust install and on PATH), else
# ~/.local/bin.  Override with BINDIR=...
if [ -z "${BINDIR:-}" ]; then
    if [ -d "$HOME/.cargo/bin" ]; then BINDIR="$HOME/.cargo/bin"; else BINDIR="$HOME/.local/bin"; fi
fi

build=1
for a in "$@"; do
    case "$a" in
        --no-build) build=0 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "install.sh: unknown flag $a" >&2; exit 2 ;;
    esac
done

if [ "$build" = 1 ]; then
    echo "building release (maka + makac)..."
    cargo build --release --manifest-path "$here/Cargo.toml" -p maka_cli -p maka_driver
fi

mkdir -p "$BINDIR"
for bin in maka makac; do
    src="$here/target/release/$bin"
    if [ ! -x "$src" ]; then
        echo "install.sh: $src not found (run without --no-build)" >&2
        exit 1
    fi
    install -m 0755 "$src" "$BINDIR/$bin"
    echo "installed $BINDIR/$bin"
done

echo
if printf '%s' ":$PATH:" | grep -q ":$BINDIR:"; then
    echo "done. Open a new shell (or rehash) and run:  maka --help"
else
    echo "done, but $BINDIR is not on your PATH. Add this to your shell profile:"
    echo "    export PATH=\"$BINDIR:\$PATH\""
fi
