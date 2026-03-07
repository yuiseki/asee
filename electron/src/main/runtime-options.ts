export type RuntimeOptions = {
  title: string;
  backendBaseUrl: string;
  pollIntervalMs: number;
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

function normalizeBaseUrl(value: string | undefined): string {
  const trimmed = value?.trim();
  if (!trimmed) {
    return 'http://127.0.0.1:8765';
  }
  return trimmed.replace(/\/+$/, '');
}

function parsePositiveInt(value: string | undefined, fallback: number): number {
  const parsed = Number.parseInt(value ?? '', 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

export function parseRuntimeOptions({ argv, env }: ParseInput): RuntimeOptions {
  return {
    title: env.ASEE_VIEWER_TITLE?.trim() || 'ASEE Viewer',
    backendBaseUrl: normalizeBaseUrl(env.ASEE_VIEWER_BACKEND_URL),
    pollIntervalMs: parsePositiveInt(env.ASEE_VIEWER_POLL_INTERVAL_MS, 2000),
    autoDemo: argv.includes('--auto-demo') || parseBool(env.ASEE_VIEWER_AUTODEMO),
    exitAfterDemo:
      argv.includes('--exit-after-demo') || parseBool(env.ASEE_VIEWER_EXIT_AFTER_DEMO),
  };
}
