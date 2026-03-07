import { contextBridge } from 'electron';

contextBridge.exposeInMainWorld('aseeViewer', {
  ready: true,
});
