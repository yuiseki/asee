export type RuntimeOptions = {
  title: string;
  autoDemo: boolean;
  exitAfterDemo: boolean;
};

type ParseInput = {
  argv: string[];
  env: Record<string, string | undefined>;
};

function parseBool(value: string | undefined): boolean {
  const lowered = value?.trim().toLowerCase();
  return lowered === '1' || lowered === 'true' || lowered === 'yes' || lowered === 'on';
}

export function parseRuntimeOptions({ argv, env }: ParseInput): RuntimeOptions {
  return {
    title: env.ASEE_VIEWER_TITLE?.trim() || 'ASEE Viewer',
    autoDemo: argv.includes('--auto-demo') || parseBool(env.ASEE_VIEWER_AUTODEMO),
    exitAfterDemo:
      argv.includes('--exit-after-demo') || parseBool(env.ASEE_VIEWER_EXIT_AFTER_DEMO),
  };
}
