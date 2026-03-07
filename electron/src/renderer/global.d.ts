import type { ViewerApi } from '@/shared/viewer-api';

declare global {
  interface Window {
    aseeViewerApi?: ViewerApi;
  }
}

export {};
