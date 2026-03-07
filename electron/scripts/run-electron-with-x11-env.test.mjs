// @vitest-environment node

import { describe, expect, it } from 'vitest';

import {
  buildElectronArgs,
  parseArgs,
  parseBool,
  parseExtraArgs,
  resolveLaunchOptions,
} from './launch-options.mjs';

describe('parseArgs', () => {
  it('recognizes viewer launch flags', () => {
    const options = parseArgs([
      '--skip-build',
      '--disable-gpu',
      '--disable-gpu-sandbox',
      '--use-gl',
      'desktop',
      '--use-angle',
      'gl',
      '--extra-arg',
      '--enable-logging=stderr',
      '--extra-arg',
      '--v=1',
      '--auto-demo',
      '--exit-after-demo',
    ]);

    expect(options).toEqual({
      autoDemo: true,
      disableGpu: true,
      disableGpuSandbox: true,
      exitAfterDemo: true,
      extraArgs: ['--enable-logging=stderr', '--v=1'],
      skipBuild: true,
      useAngle: 'gl',
      useGl: 'desktop',
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
  it('adds GPU flags only when requested', () => {
    expect(
      buildElectronArgs({
        disableGpu: false,
        disableGpuSandbox: false,
        extraArgs: [],
        useAngle: '',
        useGl: '',
      }),
    ).toEqual(['--no-sandbox', '.']);
    expect(
      buildElectronArgs({
        disableGpu: true,
        disableGpuSandbox: true,
        extraArgs: ['--enable-logging=stderr', '--v=1'],
        useAngle: 'gl',
        useGl: 'desktop',
      }),
    ).toEqual([
      '--no-sandbox',
      '--disable-gpu',
      '--disable-gpu-sandbox',
      '--use-gl=desktop',
      '--use-angle=gl',
      '--enable-logging=stderr',
      '--v=1',
      '.',
    ]);
  });
});

describe('parseExtraArgs', () => {
  it('splits whitespace-delimited extra args and trims quotes', () => {
    expect(
      parseExtraArgs(`--enable-logging=stderr --v=1 "--ignore-gpu-blocklist"`),
    ).toEqual(['--enable-logging=stderr', '--v=1', '--ignore-gpu-blocklist']);
  });
});

describe('resolveLaunchOptions', () => {
  it('merges env defaults with cli overrides', () => {
    expect(
      resolveLaunchOptions({
        cliOptions: parseArgs(['--use-gl', 'desktop', '--extra-arg', '--v=1']),
        env: {
          ASEE_VIEWER_DISABLE_GPU: '1',
          ASEE_VIEWER_DISABLE_GPU_SANDBOX: '1',
          ASEE_VIEWER_EXTRA_ARGS: '--enable-logging=stderr',
          ASEE_VIEWER_USE_ANGLE: 'gl',
          ASEE_VIEWER_USE_GL: 'egl',
        },
      }),
    ).toEqual({
      disableGpu: true,
      disableGpuSandbox: true,
      extraArgs: ['--enable-logging=stderr', '--v=1'],
      useAngle: 'gl',
      useGl: 'desktop',
    });
  });
});
