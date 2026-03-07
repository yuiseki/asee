import { buildMainWindowOptions } from './window-options';

describe('buildMainWindowOptions', () => {
  it('creates a frameless viewer window with the expected desktop defaults', () => {
    const options = buildMainWindowOptions({
      title: 'ASEE Viewer',
      preloadPath: '/tmp/preload.js',
    });

    expect(options).toMatchObject({
      width: 1600,
      height: 900,
      backgroundColor: '#07110d',
      title: 'ASEE Viewer',
      autoHideMenuBar: true,
      frame: false,
      webPreferences: {
        preload: '/tmp/preload.js',
        contextIsolation: true,
        nodeIntegration: false,
      },
    });
  });
});
