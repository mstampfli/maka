#!/usr/bin/env bash
# Build a self-contained Maka VS Code extension: compile the language server in
# release, bundle it into bin/, and produce a .vsix installable from the GUI.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../.." && pwd)"

echo "building maka-lsp (release)..."
cargo build --release -p maka_lsp --manifest-path "$root/Cargo.toml"

echo "bundling server..."
mkdir -p "$here/bin"
cp "$root/target/release/maka-lsp" "$here/bin/maka-lsp"
chmod +x "$here/bin/maka-lsp"

echo "installing client deps..."
(cd "$here" && npm install --silent)

echo "packaging .vsix..."
(cd "$here" && npx --yes @vscode/vsce package --allow-missing-repository --skip-license)

echo
echo "done. Install it from the Extensions view (... -> Install from VSIX...):"
ls -1 "$here"/*.vsix
