#!/usr/bin/env bash
# Windows test runner: cross-compile each test via MSYS2 mingw-w64 from WSL,
# run the resulting .exe under cmd.exe, capture compiler warnings + binary
# output and diff against the .expected file (same semantics as run_all.sh).
set -u
cd "$(dirname "$0")/.."
cargo build --release --quiet || exit 1
MAKAC=./target/release/makac
GCC_WIN="C:\\msys64\\mingw64\\bin\\gcc.exe"
TEMP="/mnt/c/temp"
mkdir -p "$TEMP"

pass=0
fail=0
fail_names=()

for src in tests/programs/*.maka; do
    name=$(basename "$src" .maka)
    [[ "$name" == neg_* ]] && continue
    expected="tests/programs/${name}.expected"
    [ -f "$expected" ] || continue

    # Skip rblock tests on Windows — they need rustc + the Maka rust bridge,
    # which we don't yet cross-compile from WSL.
    if grep -q '^rblock\|^rdep ' "$src"; then continue; fi
    # Skip tests with .deps (multi-module) for now — the makac --emit-c flow
    # would need to know to include the dep files.
    [ -f "tests/programs/${name}.deps" ] && continue
    # Skip explicitly linux-only tests on Windows.
    if grep -q "MAKA_TEST: linux-only" "$src"; then continue; fi

    # Compile to C, capture sema warnings.
    cfile="${TEMP}/w_${name}.c"
    exe_w="C:\\temp\\w_${name}.exe"
    exe_p="${TEMP}/w_${name}.exe"
    cflags=""
    extra_c_win=""   # extra C sources to pass to gcc (Windows path form)
    extra_link_libs=""
    link_file="tests/programs/${name}.link"
    if [ -f "$link_file" ]; then
        while IFS= read -r line; do
            [ -z "$line" ] && continue
            case "$line" in
                -l*) extra_link_libs="$extra_link_libs $line" ;;
                -L*) extra_link_libs="$extra_link_libs $line" ;;
                *)
                    cflags="$cflags --link tests/programs/$line"
                    # Copy helper .c to /mnt/c/temp so cmd.exe can reach it.
                    helper_src="tests/programs/$line"
                    helper_base=$(basename "$line")
                    cp "$helper_src" "${TEMP}/helper_${name}_${helper_base}"
                    extra_c_win="$extra_c_win C:\\\\temp\\\\helper_${name}_${helper_base}"
                    ;;
            esac
        done < "$link_file"
    fi
    sema_out=$("$MAKAC" "$src" $cflags --emit-c -o "$cfile" 2>&1)

    # Cross-compile.
    if ! cmd.exe /c "$GCC_WIN -O0 -w $(echo $cfile | sed 's|/mnt/c|C:|; s|/|\\\\|g')$extra_c_win -o $exe_w -lpthread -lws2_32 -lwinmm$extra_link_libs" 2>/dev/null; then
        :
    fi
    if [ ! -f "$exe_p" ]; then
        echo "FAIL $name (build)"
        fail=$((fail+1))
        fail_names+=("$name")
        continue
    fi

    # Run.  stdin redirect for tests that have a .stdin file — copy the
    # input file into TEMP so cmd.exe can resolve it as a plain C:\\temp path.
    stdin_file="tests/programs/${name}.stdin"
    if [ -f "$stdin_file" ]; then
        cp "$stdin_file" "${TEMP}/stdin_${name}.in"
        bin_out=$(cd "$TEMP" && timeout 15 cmd.exe /c "w_${name}.exe < stdin_${name}.in" 2>&1)
    else
        bin_out=$(cd "$TEMP" && timeout 15 cmd.exe /c "w_${name}.exe" 2>&1)
    fi

    # Combine sema warnings + binary output, matching Linux --run's 2>&1.
    if [ -n "$sema_out" ]; then
        combined="$sema_out"$'\n'"$bin_out"
    else
        combined="$bin_out"
    fi

    if [ "$combined" = "$(cat "$expected")" ]; then
        pass=$((pass+1))
    else
        echo "FAIL $name"
        fail=$((fail+1))
        fail_names+=("$name")
    fi
done

echo "windows: pass=$pass fail=$fail"
if [ $fail -gt 0 ]; then
    echo "failed:"
    for n in "${fail_names[@]}"; do echo "    $n"; done
fi
exit $([ $fail -eq 0 ] && echo 0 || echo 1)
