import { spawn } from 'node:child_process';
import { join } from 'node:path';

import { buildElectronLaunchEnv, discoverDesktopSessionEnv } from './x11-env.mjs';

function parseArgs(argv) {
  return {
    autoDemo: argv.includes('--auto-demo'),
    exitAfterDemo: argv.includes('--exit-after-demo'),
  };
}

function runCommand(command, args, options = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, {
      stdio: 'inherit',
      ...options,
    });

    child.on('error', reject);
    child.on('exit', (code, signal) => {
      if (signal) {
        reject(new Error(`${command} exited with signal ${signal}`));
        return;
      }
      if (code !== 0) {
        reject(new Error(`${command} exited with code ${code}`));
        return;
      }
      resolve();
    });
  });
}

async function main() {
  const { autoDemo, exitAfterDemo } = parseArgs(process.argv.slice(2));
  const projectRoot = join(import.meta.dirname, '..');
  const electronBinary = join(
    projectRoot,
    'node_modules',
    'electron',
    'dist',
    'electron',
  );

  const detectedEnv = await discoverDesktopSessionEnv();
  const env = buildElectronLaunchEnv({
    inheritedEnv: process.env,
    detectedEnv,
  });

  await runCommand('npm', ['run', 'build'], {
    cwd: projectRoot,
    env,
  });

  const electronEnv = {
    ...env,
    ...(autoDemo ? { ASEE_VIEWER_AUTODEMO: '1' } : {}),
    ...(exitAfterDemo ? { ASEE_VIEWER_EXIT_AFTER_DEMO: '1' } : {}),
  };

  await runCommand(electronBinary, ['.', '--no-sandbox'], {
    cwd: projectRoot,
    env: electronEnv,
  });
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
