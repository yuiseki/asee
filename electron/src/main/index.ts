import { app, BrowserWindow } from 'electron';
import { join } from 'node:path';

import { parseRuntimeOptions } from './runtime-options';

function createMainWindow(title: string): BrowserWindow {
  const window = new BrowserWindow({
    width: 1600,
    height: 900,
    backgroundColor: '#07110d',
    title,
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
    },
  });

  const rendererUrl = process.env.VITE_DEV_SERVER_URL;
  if (rendererUrl) {
    void window.loadURL(rendererUrl);
  } else {
    void window.loadFile(join(__dirname, '../renderer/index.html'));
  }

  return window;
}

const runtimeOptions = parseRuntimeOptions({
  argv: process.argv,
  env: process.env,
});

app.whenReady().then(() => {
  createMainWindow(runtimeOptions.title);

  if (runtimeOptions.autoDemo && runtimeOptions.exitAfterDemo) {
    setTimeout(() => {
      void app.quit();
    }, 1500);
  }

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow(runtimeOptions.title);
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    void app.quit();
  }
});
