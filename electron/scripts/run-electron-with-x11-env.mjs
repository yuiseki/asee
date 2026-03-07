import { spawn } from 'node:child_process';
import { join } from 'node:path';

import { buildElectronArgs, parseArgs, resolveLaunchOptions } from './launch-options.mjs';
import { buildElectronLaunchEnv, discoverDesktopSessionEnv } from './x11-env.mjs';

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
  const cliOptions = parseArgs(process.argv.slice(2));
  const { autoDemo, exitAfterDemo, skipBuild } = cliOptions;
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

  if (!skipBuild) {
    await runCommand('npm', ['run', 'build'], {
      cwd: projectRoot,
      env,
    });
  }

  const electronEnv = {
    ...env,
    ...(autoDemo ? { ASEE_VIEWER_AUTODEMO: '1' } : {}),
    ...(exitAfterDemo ? { ASEE_VIEWER_EXIT_AFTER_DEMO: '1' } : {}),
  };
  const launchOptions = resolveLaunchOptions({
    cliOptions,
    env: process.env,
  });
  const electronArgs = buildElectronArgs(launchOptions);
  console.error(
    `ASEE viewer launch args: ${electronArgs.join(' ')}`,
  );

  await runCommand(
    electronBinary,
    electronArgs,
    {
      cwd: projectRoot,
      env: electronEnv,
    },
  );
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : String(error));
  process.exit(1);
});
