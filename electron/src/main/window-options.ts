import type { BrowserWindowConstructorOptions } from 'electron';

type BuildMainWindowOptionsInput = {
  title: string;
  preloadPath: string;
};

export function buildMainWindowOptions({
  title,
  preloadPath,
}: BuildMainWindowOptionsInput): BrowserWindowConstructorOptions {
  return {
    width: 1600,
    height: 900,
    backgroundColor: '#07110d',
    title,
    autoHideMenuBar: true,
    frame: false,
    webPreferences: {
      preload: preloadPath,
      contextIsolation: true,
      nodeIntegration: false,
    },
  };
}
