type StartupShowWindow = {
  once(event: 'ready-to-show', listener: () => void): void;
  showInactive(): void;
};

export function bindStartupShowInactive(window: StartupShowWindow): void {
  window.once('ready-to-show', () => {
    window.showInactive();
  });
}
