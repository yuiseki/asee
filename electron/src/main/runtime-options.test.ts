import { parseRuntimeOptions } from './runtime-options';

describe('parseRuntimeOptions', () => {
  it('uses defaults when no flags or env overrides are present', () => {
    const options = parseRuntimeOptions({
      argv: ['node', 'electron'],
      env: {},
    });

    expect(options.title).toBe('ASEE Viewer');
    expect(options.backendBaseUrl).toBe('http://127.0.0.1:8765');
    expect(options.pollIntervalMs).toBe(2000);
    expect(options.autoDemo).toBe(false);
    expect(options.exitAfterDemo).toBe(false);
  });

  it('accepts env and argv overrides for demo mode', () => {
    const options = parseRuntimeOptions({
      argv: ['node', 'electron', '--auto-demo'],
      env: {
        ASEE_VIEWER_TITLE: 'ASEE Camera Grid',
        ASEE_VIEWER_BACKEND_URL: 'http://127.0.0.1:18766/',
        ASEE_VIEWER_POLL_INTERVAL_MS: '1500',
        ASEE_VIEWER_EXIT_AFTER_DEMO: '1',
      },
    });

    expect(options.title).toBe('ASEE Camera Grid');
    expect(options.backendBaseUrl).toBe('http://127.0.0.1:18766');
    expect(options.pollIntervalMs).toBe(1500);
    expect(options.autoDemo).toBe(true);
    expect(options.exitAfterDemo).toBe(true);
  });
});
