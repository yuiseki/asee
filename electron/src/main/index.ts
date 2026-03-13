import { app, BrowserWindow, screen } from 'electron';
import { join } from 'node:path';

import { parseRuntimeOptions } from './runtime-options';
import { buildDefaultViewerBounds, buildMainWindowOptions } from './window-options';
import { bindStartupShowInactive } from './window-show';

function createMainWindow(title: string): BrowserWindow {
  const bounds = buildDefaultViewerBounds(screen.getPrimaryDisplay().workArea);
  const window = new BrowserWindow(
    buildMainWindowOptions({
      title,
      preloadPath: join(__dirname, '../preload/index.js'),
      bounds,
    }),
  );

  const rendererUrl = process.env.VITE_DEV_SERVER_URL;
  if (rendererUrl) {
    void window.loadURL(rendererUrl);
  } else {
    void window.loadFile(join(__dirname, '../renderer/index.html'));
  }
  bindStartupShowInactive(window);

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
