export type ViewerStatus = {
  running: boolean;
};

export type OverlayText = {
  caption: string;
  prediction: string;
};

export type BiometricStatus = {
  running: boolean;
  ownerEmbeddingLoaded: boolean;
  ownerPresent: boolean;
  ownerCount: number;
  subjectCount: number;
  peopleCount: number;
  ownerSeenAgoMs: number | null;
  updatedAt: number;
};

export type ViewerSnapshot = {
  baseUrl: string;
  cameras: number[];
  status: ViewerStatus;
  overlayText: OverlayText;
  biometricStatus: BiometricStatus;
};

export type ViewerConfig = {
  title: string;
  backendBaseUrl: string;
  pollIntervalMs: number;
  autoDemo: boolean;
};

export type ViewerApi = {
  getConfig: () => ViewerConfig;
  fetchSnapshot: () => Promise<ViewerSnapshot>;
};
