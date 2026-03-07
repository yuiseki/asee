const DESKTOP_PROCESS_PRIORITY = [
  'plasmashell',
  'kwin_x11',
  'kwin_wayland',
  'startplasma-x11',
  'startplasma-wayland',
  'Xorg',
  'Xwayland',
];

function normalizeEnvValue(value) {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined;
}

export function parseProcEnviron(raw) {
  const env = {};
  for (const entry of String(raw).split('\0')) {
    if (!entry) {
      continue;
    }
    const separatorIndex = entry.indexOf('=');
    if (separatorIndex <= 0) {
      continue;
    }
    env[entry.slice(0, separatorIndex)] = entry.slice(separatorIndex + 1);
  }
  return env;
}

export function pickDesktopSessionEnv(candidates) {
  const sortedCandidates = [...candidates].sort((left, right) => {
    const leftPriority = DESKTOP_PROCESS_PRIORITY.indexOf(left.processName);
    const rightPriority = DESKTOP_PROCESS_PRIORITY.indexOf(right.processName);
    const leftScore = leftPriority >= 0 ? leftPriority : DESKTOP_PROCESS_PRIORITY.length;
    const rightScore = rightPriority >= 0 ? rightPriority : DESKTOP_PROCESS_PRIORITY.length;
    return leftScore - rightScore;
  });

  for (const candidate of sortedCandidates) {
    const display = normalizeEnvValue(candidate.env.DISPLAY);
    const xauthority = normalizeEnvValue(candidate.env.XAUTHORITY);
    if (display && xauthority) {
      return {
        DISPLAY: display,
        XAUTHORITY: xauthority,
      };
    }
  }

  return null;
}

export function buildElectronLaunchEnv({ inheritedEnv, detectedEnv }) {
  return {
    ...inheritedEnv,
    DISPLAY:
      normalizeEnvValue(inheritedEnv.ASEE_VIEWER_DISPLAY) ??
      normalizeEnvValue(inheritedEnv.DISPLAY) ??
      normalizeEnvValue(detectedEnv?.DISPLAY) ??
      ':0',
    XAUTHORITY:
      normalizeEnvValue(inheritedEnv.ASEE_VIEWER_XAUTHORITY) ??
      normalizeEnvValue(inheritedEnv.XAUTHORITY) ??
      normalizeEnvValue(detectedEnv?.XAUTHORITY) ??
      `${inheritedEnv.HOME ?? ''}/.Xauthority`,
  };
}

export { DESKTOP_PROCESS_PRIORITY };
