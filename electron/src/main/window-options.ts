import type { BrowserWindowConstructorOptions } from 'electron';

type ViewerBounds = {
  x: number;
  y: number;
  width: number;
  height: number;
};

type BuildMainWindowOptionsInput = {
  title: string;
  preloadPath: string;
  bounds?: ViewerBounds;
};

export function buildDefaultViewerBounds(workArea: ViewerBounds): ViewerBounds {
  const width = Math.floor(workArea.width / 2);
  const height = Math.floor(workArea.height / 2);
  return {
    x: workArea.x,
    y: workArea.y + (workArea.height - height),
    width,
    height,
  };
}

export function buildMainWindowOptions({
  title,
  preloadPath,
  bounds,
}: BuildMainWindowOptionsInput): BrowserWindowConstructorOptions {
  return {
    x: bounds?.x,
    y: bounds?.y,
    width: bounds?.width ?? 1600,
    height: bounds?.height ?? 900,
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
