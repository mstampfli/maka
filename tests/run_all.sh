#!/usr/bin/env bash
# Compile and run every program in tests/programs/, checking stdout vs the .expected file.
set -u
cd "$(dirname "$0")/.."
cargo build --quiet || exit 1
MAKAC=./target/debug/makac
fail=0
for src in tests/programs/*.maka; do
    name=$(basename "$src" .maka)
    [[ "$name" == neg_* ]] && continue
    expected="tests/programs/${name}.expected"
    [ -f "$expected" ] || continue
    bin="/tmp/maka_${name}"
    extras=""
    deps_file="tests/programs/${name}.deps"
    if [ -f "$deps_file" ]; then
        while IFS= read -r line; do
            [ -n "$line" ] && extras="$extras tests/programs/$line"
        done < "$deps_file"
    fi
    cflags=""
    link_file="tests/programs/${name}.link"
    if [ -f "$link_file" ]; then
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            case "$line" in
                -l*|-L*) cflags="$cflags --link $line" ;;
                *)       cflags="$cflags --link tests/programs/$line" ;;
            esac
        done < "$link_file"
    fi
    stdin_file="tests/programs/${name}.stdin"
    if [ -f "$stdin_file" ]; then
        out=$("$MAKAC" $extras "$src" $cflags -o "$bin" --run < "$stdin_file" 2>&1)
    else
        out=$("$MAKAC" $extras "$src" $cflags -o "$bin" --run 2>&1)
    fi
    rc=$?
    if [ "$rc" != "0" ]; then
        echo "FAIL $name: exit $rc"
        echo "$out" | sed 's/^/    /'
        fail=1
        continue
    fi
    if [ "$out" = "$(cat "$expected")" ]; then
        # Optional emitted-C assertions (e.g. that devirtualization fired): a
        # `NN.cgrep` sidecar with one fixed-string pattern per line - `!pattern`
        # must be ABSENT, `pattern` must be PRESENT, `#` lines are comments.
        cgrep_file="tests/programs/${name}.cgrep"
        if [ -f "$cgrep_file" ]; then
            cfile="/tmp/maka_${name}.c"
            "$MAKAC" $extras "$src" $cflags --emit-c -o "$cfile" >/dev/null 2>&1
            cg_fail=0
            while IFS= read -r pat; do
                case "$pat" in
                    ""|"#"*) ;;
                    "!"*) grep -qF -- "${pat#!}" "$cfile" && { echo "FAIL $name (emitted C must NOT contain: ${pat#!})"; cg_fail=1; } ;;
                    *)    grep -qF -- "$pat" "$cfile" || { echo "FAIL $name (emitted C missing: $pat)"; cg_fail=1; } ;;
                esac
            done < "$cgrep_file"
            if [ "$cg_fail" != "0" ]; then fail=1; continue; fi
        fi
        echo "ok   $name"
    else
        echo "FAIL $name (output mismatch)"
        diff <(echo "$out") "$expected" | sed 's/^/    /'
        fail=1
    fi
done
exit "$fail"
