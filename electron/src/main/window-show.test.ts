import { bindStartupShowInactive } from './window-show';

describe('bindStartupShowInactive', () => {
  it('shows the window without stealing focus when ready', () => {
    let readyToShowHandler: (() => void) | undefined;
    const window = {
      once: (event: string, handler: () => void) => {
        expect(event).toBe('ready-to-show');
        readyToShowHandler = handler;
      },
      showInactive: vi.fn(),
    };

    bindStartupShowInactive(window);
    readyToShowHandler?.();

    expect(window.showInactive).toHaveBeenCalledTimes(1);
  });
});
