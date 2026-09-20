// frontend/src/main/index.js
// Electron main process entry point.

const { app, BrowserWindow } = require('electron');
const path = require('path');

const DEV_SERVER_URL = process.env.ELECTRON_START_URL;

let mainWindow = null;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  if (!app.isPackaged && DEV_SERVER_URL) {
    // Development: load the dev server (e.g. Vite on http://localhost:5173).
    mainWindow.loadURL(DEV_SERVER_URL);
  } else {
    // Production: load the bundled renderer HTML.
    mainWindow.loadFile(path.join(__dirname, '..', 'renderer', 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    // macOS: re-create a window when the dock icon is clicked and none are open.
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});