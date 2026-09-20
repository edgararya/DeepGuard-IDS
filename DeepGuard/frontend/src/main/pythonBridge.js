// frontend/src/main/pythonBridge.js
// Manages the PyInstaller-packaged Python backend as a child process.

const { app } = require('electron');
const { spawn, execFileSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Matches the readiness line printed by backend/app/main.py:
//   [Xeptocore] Backend ready on port 8000
const PORT_PATTERN = /Backend ready on port\s+(\d+)/;

// The backend picks its own free port, so we wait for it to report that port.
const STARTUP_TIMEOUT_MS = 30000;

let backendProcess = null;
let backendPort = null;
let startupTimeout = null;
let startupTimeoutHandler = null;

function backendExecutablePath() {
  const binary = process.platform === 'win32' ? 'backend.exe' : 'backend';
  const platformDir = process.platform === 'win32'
    ? 'windows'
    : process.platform === 'darwin'
      ? 'macos'
      : 'linux';
  return path.join(__dirname, 'backend', platformDir, binary);
}

function clearStartupTimeout() {
  if (startupTimeout !== null) {
    clearTimeout(startupTimeout);
    startupTimeout = null;
  }
}

function startBackend(onStartupTimeout) {
  if (backendProcess) {
    return; // already running
  }

  clearStartupTimeout();

  if (typeof onStartupTimeout === 'function') {
    startupTimeoutHandler = onStartupTimeout;
  }

  const executable = backendExecutablePath();
  if (!fs.existsSync(executable)) {
    console.error(`[pythonBridge] Backend executable not found: ${executable}`);
    return;
  }

  // No PORT is passed: the backend selects an available port itself and reports it on stdout.
  backendProcess = spawn(executable, [], {
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: process.platform !== 'win32',
  });

  // Fail fast if the backend never reports a port (e.g. it hangs while loading ML models).
  startupTimeout = setTimeout(() => {
    startupTimeout = null;
    if (backendPort !== null) {
      return; // it did become ready after all
    }
    const message = `[pythonBridge] Backend did not become ready within ${STARTUP_TIMEOUT_MS / 1000}s`;
    console.error(message);
    if (typeof startupTimeoutHandler === 'function') {
      try {
        startupTimeoutHandler(new Error(message));
      } catch (err) {
        console.error(`[pythonBridge] onStartupTimeout callback failed: ${err.message}`);
      }
    }
  }, STARTUP_TIMEOUT_MS);

  let buffer = '';
  backendProcess.stdout.on('data', (chunk) => {
    buffer += chunk.toString();
    let newlineIndex;
    while ((newlineIndex = buffer.indexOf('\n')) !== -1) {
      const line = buffer.slice(0, newlineIndex).trim();
      buffer = buffer.slice(newlineIndex + 1);

      console.log(`[backend] ${line}`);

      const match = line.match(PORT_PATTERN);
      if (match && backendPort === null) {
        backendPort = parseInt(match[1], 10);
        clearStartupTimeout();
        console.log(`[pythonBridge] Backend ready on port ${backendPort}`);
      }
    }
  });

  backendProcess.stderr.on('data', (chunk) => {
    console.error(`[backend:stderr] ${chunk.toString()}`);
  });

  backendProcess.on('error', (err) => {
    console.error(`[pythonBridge] Failed to start backend: ${err.message}`);
    clearStartupTimeout();
    backendProcess = null;
  });

  backendProcess.on('exit', (code, signal) => {
    console.log(`[pythonBridge] Backend exited (code=${code}, signal=${signal})`);
    clearStartupTimeout();
    backendProcess = null;
    backendPort = null;
  });
}

function getBackendPort() {
  return backendPort;
}

function isBackendReady() {
  return backendPort !== null;
}

function stopBackend() {
  if (!backendProcess) {
    return;
  }

  const proc = backendProcess;
  backendProcess = null;
  backendPort = null;
  clearStartupTimeout();

  try {
    if (process.platform === 'win32') {
      // /T kills the whole process tree; /F forces. Required for PyInstaller one-file builds.
      execFileSync('taskkill', ['/pid', String(proc.pid), '/T', '/F']);
    } else {
      // Negative PID kills the entire detached process group.
      process.kill(-proc.pid, 'SIGTERM');
    }
  } catch (err) {
    console.error(`[pythonBridge] Error stopping backend: ${err.message}`);
  }
}

function initBackendBridge(options = {}) {
  const { onStartupTimeout } = options;
  app.whenReady().then(() => startBackend(onStartupTimeout));
  app.on('before-quit', stopBackend);
}

module.exports = {
  startBackend,
  getBackendPort,
  isBackendReady,
  stopBackend,
  initBackendBridge,
};