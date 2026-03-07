export function parseArgs(argv) {
  const options = {
    autoDemo: false,
    disableGpu: false,
    disableGpuSandbox: false,
    exitAfterDemo: false,
    extraArgs: [],
    skipBuild: false,
    useAngle: '',
    useGl: '',
  };

  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    switch (arg) {
      case '--auto-demo':
        options.autoDemo = true;
        break;
      case '--exit-after-demo':
        options.exitAfterDemo = true;
        break;
      case '--skip-build':
        options.skipBuild = true;
        break;
      case '--disable-gpu':
        options.disableGpu = true;
        break;
      case '--disable-gpu-sandbox':
        options.disableGpuSandbox = true;
        break;
      case '--use-gl':
        index += 1;
        options.useGl = argv[index] ?? '';
        break;
      case '--use-angle':
        index += 1;
        options.useAngle = argv[index] ?? '';
        break;
      case '--extra-arg':
        index += 1;
        if (argv[index]) {
          options.extraArgs.push(argv[index]);
        }
        break;
      default:
        break;
    }
  }

  return options;
}

export function parseBool(value) {
  const lowered = value?.trim().toLowerCase();
  return lowered === '1' || lowered === 'true' || lowered === 'yes' || lowered === 'on';
}

export function parseExtraArgs(value) {
  if (!value?.trim()) {
    return [];
  }

  const parts = value.match(/"[^"]*"|'[^']*'|[^\s]+/g) ?? [];
  return parts.map((part) => {
    if (
      (part.startsWith('"') && part.endsWith('"'))
      || (part.startsWith("'") && part.endsWith("'"))
    ) {
      return part.slice(1, -1);
    }
    return part;
  });
}

export function resolveLaunchOptions({ cliOptions, env }) {
  return {
    disableGpu: cliOptions.disableGpu || parseBool(env.ASEE_VIEWER_DISABLE_GPU),
    disableGpuSandbox:
      cliOptions.disableGpuSandbox || parseBool(env.ASEE_VIEWER_DISABLE_GPU_SANDBOX),
    extraArgs: [...parseExtraArgs(env.ASEE_VIEWER_EXTRA_ARGS), ...cliOptions.extraArgs],
    useAngle: cliOptions.useAngle || env.ASEE_VIEWER_USE_ANGLE?.trim() || '',
    useGl: cliOptions.useGl || env.ASEE_VIEWER_USE_GL?.trim() || '',
  };
}

export function buildElectronArgs({
  disableGpu,
  disableGpuSandbox,
  extraArgs = [],
  useAngle = '',
  useGl = '',
}) {
  return [
    '--no-sandbox',
    ...(disableGpu ? ['--disable-gpu'] : []),
    ...(disableGpuSandbox ? ['--disable-gpu-sandbox'] : []),
    ...(useGl ? [`--use-gl=${useGl}`] : []),
    ...(useAngle ? [`--use-angle=${useAngle}`] : []),
    ...extraArgs,
    '.',
  ];
}
