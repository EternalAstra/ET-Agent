/**
 * ET-Agent Memory Monitor — Electron Desktop App
 *
 * Opens a native window that connects to the local monitor API server
 * (scripts/monitor_api.py) and displays real-time KV Cache memory stats.
 *
 * Usage:
 *   npm install && npm start
 *
 * Requires: python scripts/monitor_api.py running on localhost:8765
 */

const { app, BrowserWindow, Menu, Tray, nativeImage, dialog } = require("electron");
const path = require("path");
const { exec } = require("child_process");

const API_PORT = process.env.ET_MONITOR_PORT || 8765;
const DASHBOARD_URL = `http://localhost:${API_PORT}`;

let mainWindow = null;
let tray = null;

// ── Create the main window ────────────────────────────────────────
function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 700,
    title: "ET-Agent Memory Monitor",
    backgroundColor: "#0f1117",
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
    icon: undefined, // set your .ico path here if desired
  });

  mainWindow.loadURL(DASHBOARD_URL);

  mainWindow.on("closed", () => {
    mainWindow = null;
  });

  // Dev: open DevTools
  if (process.argv.includes("--dev")) {
    mainWindow.webContents.openDevTools();
  }
}

// ── System tray ───────────────────────────────────────────────────
function createTray() {
  // Create a simple 16x16 icon programmatically
  const icon = nativeImage.createFromDataURL(
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAARklEQVQ4y2Ng+M9AAWBiYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGBgYGD4TwFgAABJ9gHr+6XLnAAAAABJRU5ErkJggg=="
  );
  tray = new Tray(icon);
  const contextMenu = Menu.buildFromTemplate([
    { label: "Show Monitor", click: () => mainWindow && mainWindow.show() },
    { type: "separator" },
    { label: "Quit", click: () => app.quit() },
  ]);
  tray.setToolTip("ET-Agent Memory Monitor");
  tray.setContextMenu(contextMenu);
  tray.on("click", () => {
    if (mainWindow) {
      mainWindow.isVisible() ? mainWindow.hide() : mainWindow.show();
    }
  });
}

// ── Auto-start Python API server ──────────────────────────────────
function startApiServer() {
  const scriptPath = path.join(__dirname, "..", "..", "..", "scripts", "monitor_api.py");
  const pythonCmd = process.platform === "win32" ? "python" : "python3";

  const proc = exec(
    `${pythonCmd} "${scriptPath}" --port ${API_PORT} --init --gpu-gb 6`,
    { cwd: path.join(__dirname, "..", "..", "..") },
    (error, stdout, stderr) => {
      if (error && !error.killed) {
        console.error("[monitor] API server exited:", error.message);
      }
    }
  );

  proc.stdout?.on("data", (d) => process.stdout.write(`[api] ${d}`));
  proc.stderr?.on("data", (d) => process.stderr.write(`[api:err] ${d}`));

  // Give the API server time to start
  return new Promise((resolve) => setTimeout(resolve, 2000));
}

// ── App lifecycle ─────────────────────────────────────────────────
app.whenReady().then(async () => {
  // Try to start the Python API server
  try {
    await startApiServer();
  } catch (e) {
    console.error("Failed to start API server:", e.message);
  }

  createWindow();
  createTray();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on("window-all-closed", () => {
  // Keep running in tray on macOS
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", () => {
  // Cleanup
});
