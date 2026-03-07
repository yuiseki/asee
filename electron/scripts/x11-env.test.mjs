// @vitest-environment node

import { describe, expect, it } from 'vitest';

import {
  buildElectronLaunchEnv,
  parseProcEnviron,
  pickDesktopSessionEnv,
} from './x11-env-core.mjs';

describe('parseProcEnviron', () => {
  it('parses null-separated environment text', () => {
    const parsed = parseProcEnviron(
      'DISPLAY=:0\0XAUTHORITY=/run/user/1000/gdm/Xauthority\0',
    );

    expect(parsed).toEqual({
      DISPLAY: ':0',
      XAUTHORITY: '/run/user/1000/gdm/Xauthority',
    });
  });
});

describe('pickDesktopSessionEnv', () => {
  it('prefers desktop session processes over shell defaults', () => {
    const candidate = pickDesktopSessionEnv([
      {
        processName: 'bash',
        env: {
          DISPLAY: ':1',
          XAUTHORITY: '/home/yuiseki/.Xauthority',
        },
      },
      {
        processName: 'plasmashell',
        env: {
          DISPLAY: ':0',
          XAUTHORITY: '/run/user/1000/gdm/Xauthority',
        },
      },
    ]);

    expect(candidate).toEqual({
      DISPLAY: ':0',
      XAUTHORITY: '/run/user/1000/gdm/Xauthority',
    });
  });
});

describe('buildElectronLaunchEnv', () => {
  it('respects explicit ASEE overrides first', () => {
    const env = buildElectronLaunchEnv({
      inheritedEnv: {
        ASEE_VIEWER_DISPLAY: ':9',
        ASEE_VIEWER_XAUTHORITY: '/tmp/custom.Xauthority',
      },
      detectedEnv: {
        DISPLAY: ':0',
        XAUTHORITY: '/run/user/1000/gdm/Xauthority',
      },
    });

    expect(env.DISPLAY).toBe(':9');
    expect(env.XAUTHORITY).toBe('/tmp/custom.Xauthority');
  });
});
