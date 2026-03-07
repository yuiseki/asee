// @vitest-environment node

import { describe, expect, it } from 'vitest';

import {
  buildElectronArgs,
  parseArgs,
  parseBool,
} from './launch-options.mjs';

describe('parseArgs', () => {
  it('recognizes skip-build and disable-gpu flags', () => {
    const options = parseArgs([
      '--skip-build',
      '--disable-gpu',
      '--auto-demo',
      '--exit-after-demo',
    ]);

    expect(options).toEqual({
      autoDemo: true,
      disableGpu: true,
      exitAfterDemo: true,
      skipBuild: true,
    });
  });
});

describe('parseBool', () => {
  it('treats common truthy strings as enabled', () => {
    expect(parseBool('1')).toBe(true);
    expect(parseBool('true')).toBe(true);
    expect(parseBool('yes')).toBe(true);
    expect(parseBool('on')).toBe(true);
    expect(parseBool('0')).toBe(false);
  });
});

describe('buildElectronArgs', () => {
  it('adds disable-gpu only when requested', () => {
    expect(buildElectronArgs({ disableGpu: false })).toEqual(['.', '--no-sandbox']);
    expect(buildElectronArgs({ disableGpu: true })).toEqual([
      '.',
      '--no-sandbox',
      '--disable-gpu',
    ]);
  });
});
