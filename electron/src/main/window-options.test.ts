import { buildDefaultViewerBounds, buildMainWindowOptions } from './window-options';

describe('buildMainWindowOptions', () => {
  it('creates a frameless viewer window with the expected desktop defaults', () => {
    const options = buildMainWindowOptions({
      title: 'ASEE Viewer',
      preloadPath: '/tmp/preload.js',
      bounds: { x: 0, y: 1058, width: 2048, height: 1058 },
    });

    expect(options).toMatchObject({
      x: 0,
      y: 1058,
      width: 2048,
      height: 1058,
      show: false,
      backgroundColor: '#07110d',
      title: 'ASEE Viewer',
      autoHideMenuBar: true,
      frame: false,
      focusable: true,
      webPreferences: {
        preload: '/tmp/preload.js',
        contextIsolation: true,
        nodeIntegration: false,
      },
    });
  });
});

describe('buildDefaultViewerBounds', () => {
  it('targets the left-bottom quadrant of the work area', () => {
    expect(
      buildDefaultViewerBounds({
        x: 0,
        y: 0,
        width: 4096,
        height: 2116,
      }),
    ).toEqual({
      x: 0,
      y: 1058,
      width: 2048,
      height: 1058,
    });
  });
});
