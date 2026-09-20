// frontend/src/main/pythonBridge.js
// Manages the PyInstaller-packaged Python backend as a child process.

const { app } = require('electron');
const { spawn, execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

// Matches the readiness line printed by backend/app/main.py:
//   [Xeptocore] Backend ready on port 8000
const PORT_PATTERN = /Backend ready on port\s+(\d+)/;

let backendProcess = null;
let backendPort = null;

function backendExecutablePath() {
  const binary = process.platform === 'win32' ? 'backend.exe' : 'backend';
  const platformDir = process.platform === 'win32'
    ? 'windows'
    : process.platform === 'darwin'
      ? 'macos'
      : 'linux';
  return path.join(__dirname, 'backend', platformDir, binary);
}

function startBackend() {
  if (backendProcess) {
    return; // already running
  }

  const executable = backendExecutablePath();
  if (!fs.existsSync(executable)) {
    console.error(`[pythonBridge] Backend executable not found: ${executable}`);
    return;
  }

  backendProcess = spawn(executable, [], {
    env: { ...process.env, PORT: '8000' },
    stdio: ['ignore', 'pipe', 'pipe'],
    detached: process.platform !== 'win32',
  });

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
        console.log(`[pythonBridge] Backend ready on port ${backendPort}`);
      }
    }
  });

  backendProcess.stderr.on('data', (chunk) => {
    console.error(`[backend:stderr] ${chunk.toString()}`);
  });

  backendProcess.on('error', (err) => {
    console.error(`[pythonBridge] Failed to start backend: ${err.message}`);
    backendProcess = null;
  });

  backendProcess.on('exit', (code, signal) => {
    console.log(`[pythonBridge] Backend exited (code=${code}, signal=${signal})`);
    backendProcess = null;
    backendPort = null;
  });
}

function getBackendPort() {
  return backendPort;
}

function stopBackend() {
  if (!backendProcess) {
    return;
  }

  const proc = backendProcess;
  backendProcess = null;
  backendPort = null;

  try {
    if (process.platform === 'win32') {
      // /T kills the whole process tree; /F forces. Required for PyInstaller one-file builds.
      execSync(`taskkill /pid ${proc.pid} /T /F`);
    } else {
      // Negative PID kills the entire detached process group.
      process.kill(-proc.pid, 'SIGTERM');
    }
  } catch (err) {
    console.error(`[pythonBridge] Error stopping backend: ${err.message}`);
  }
}

function initBackendBridge() {
  app.whenReady().then(startBackend);
  app.on('before-quit', stopBackend);
}

module.exports = {
  startBackend,
  getBackendPort,
  stopBackend,
  initBackendBridge,
};