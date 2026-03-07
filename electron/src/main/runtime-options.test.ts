import { parseRuntimeOptions } from './runtime-options';

describe('parseRuntimeOptions', () => {
  it('uses defaults when no flags or env overrides are present', () => {
    const options = parseRuntimeOptions({
      argv: ['node', 'electron'],
      env: {},
    });

    expect(options.title).toBe('ASEE Viewer');
    expect(options.autoDemo).toBe(false);
    expect(options.exitAfterDemo).toBe(false);
  });

  it('accepts env and argv overrides for demo mode', () => {
    const options = parseRuntimeOptions({
      argv: ['node', 'electron', '--auto-demo'],
      env: {
        ASEE_VIEWER_TITLE: 'ASEE Camera Grid',
        ASEE_VIEWER_EXIT_AFTER_DEMO: '1',
      },
    });

    expect(options.title).toBe('ASEE Camera Grid');
    expect(options.autoDemo).toBe(true);
    expect(options.exitAfterDemo).toBe(true);
  });
});
