// Launches the maka-lsp language server and wires it to VS Code.  Syntax
// highlighting works with no server; this adds diagnostics, hover, go-to-
// definition, the outline, and completion when `maka-lsp` is available.

const vscode = require("vscode");
const fs = require("fs");
const path = require("path");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

let client;

function activate(context) {
  const cfg = vscode.workspace.getConfiguration("maka");
  if (!cfg.get("server.enabled", true)) {
    return;
  }
  // Prefer an explicit setting; else a server bundled in the extension (a
  // self-contained .vsix install); else `maka-lsp` on PATH.
  const bundled = path.join(
    context.extensionPath,
    "bin",
    process.platform === "win32" ? "maka-lsp.exe" : "maka-lsp"
  );
  const configured = (cfg.get("server.path") || "").trim();
  let command;
  if (configured) {
    command = configured;
  } else if (fs.existsSync(bundled)) {
    // A .vsix zip may not preserve the exec bit; restore it defensively.
    try {
      fs.chmodSync(bundled, 0o755);
    } catch (e) {
      /* ignore */
    }
    command = bundled;
  } else {
    command = "maka-lsp";
  }

  const serverOptions = {
    run: { command, transport: TransportKind.stdio },
    debug: { command, transport: TransportKind.stdio },
  };
  const clientOptions = {
    documentSelector: [{ scheme: "file", language: "maka" }],
    outputChannelName: "Maka Language Server",
  };

  client = new LanguageClient(
    "maka",
    "Maka Language Server",
    serverOptions,
    clientOptions
  );
  // If the server binary is missing, surface a friendly hint instead of a
  // silent failure.
  client.start().catch((err) => {
    vscode.window.showWarningMessage(
      `maka-lsp did not start (${command}). Set "maka.server.path" or add it to PATH. ${err}`
    );
  });
  context.subscriptions.push({ dispose: () => client && client.stop() });
}

function deactivate() {
  return client ? client.stop() : undefined;
}

module.exports = { activate, deactivate };
