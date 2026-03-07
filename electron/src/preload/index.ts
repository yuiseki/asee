import { contextBridge } from 'electron';

import type { ViewerApi } from '../shared/viewer-api';

import { parseRuntimeOptions } from '../main/runtime-options';

import { buildDemoViewerSnapshot, fetchViewerSnapshot } from './backend-client';

const runtimeOptions = parseRuntimeOptions({
  argv: process.argv,
  env: process.env,
});

const viewerApi: ViewerApi = {
  getConfig() {
    return {
      title: runtimeOptions.title,
      backendBaseUrl: runtimeOptions.backendBaseUrl,
      pollIntervalMs: runtimeOptions.pollIntervalMs,
      autoDemo: runtimeOptions.autoDemo,
    };
  },
  async fetchSnapshot() {
    if (runtimeOptions.autoDemo) {
      return buildDemoViewerSnapshot(runtimeOptions.backendBaseUrl);
    }
    return fetchViewerSnapshot({
      baseUrl: runtimeOptions.backendBaseUrl,
      fetchImpl: fetch,
    });
  },
};

contextBridge.exposeInMainWorld('aseeViewerApi', viewerApi);
