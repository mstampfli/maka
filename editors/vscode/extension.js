// Launches the maka-lsp language server and wires it to VS Code.  Syntax
// highlighting works with no server; this adds diagnostics, hover, go-to-
// definition, the outline, and completion when `maka-lsp` is available.

const vscode = require("vscode");
const { LanguageClient, TransportKind } = require("vscode-languageclient/node");

let client;

function activate(context) {
  const cfg = vscode.workspace.getConfiguration("maka");
  if (!cfg.get("server.enabled", true)) {
    return;
  }
  const command = cfg.get("server.path") || "maka-lsp";

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
