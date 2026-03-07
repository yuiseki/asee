export function parseArgs(argv) {
  return {
    autoDemo: argv.includes('--auto-demo'),
    exitAfterDemo: argv.includes('--exit-after-demo'),
    skipBuild: argv.includes('--skip-build'),
    disableGpu: argv.includes('--disable-gpu'),
  };
}

export function parseBool(value) {
  const lowered = value?.trim().toLowerCase();
  return lowered === '1' || lowered === 'true' || lowered === 'yes' || lowered === 'on';
}

export function buildElectronArgs({ disableGpu }) {
  return ['.', '--no-sandbox', ...(disableGpu ? ['--disable-gpu'] : [])];
}
