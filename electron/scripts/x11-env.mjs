import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';

import { buildElectronLaunchEnv, parseProcEnviron, pickDesktopSessionEnv } from './x11-env-core.mjs';

async function readProcEnv(pid) {
  try {
    const [comm, environ] = await Promise.all([
      readFile(join('/proc', pid, 'comm'), 'utf8'),
      readFile(join('/proc', pid, 'environ'), 'utf8'),
    ]);
    return {
      processName: comm.trim(),
      env: parseProcEnviron(environ),
    };
  } catch {
    return null;
  }
}

export async function discoverDesktopSessionEnv() {
  let entries;
  try {
    entries = await readdir('/proc');
  } catch {
    return null;
  }

  const candidates = [];
  for (const entry of entries) {
    if (!/^\d+$/.test(entry)) {
      continue;
    }
    const candidate = await readProcEnv(entry);
    if (candidate) {
      candidates.push(candidate);
    }
  }
  return pickDesktopSessionEnv(candidates);
}

export { buildElectronLaunchEnv };
