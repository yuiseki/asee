"""Static web shell helpers shared by the future Electron viewer."""

from __future__ import annotations


def build_web_manifest(title: str) -> dict[str, object]:
    """Return a minimal installable web manifest for the viewer shell."""
    return {
        "name": title,
        "short_name": "GOD MODE",
        "start_url": "/",
        "scope": "/",
        "display": "standalone",
        "background_color": "#000000",
        "theme_color": "#001a12",
        "description": "Local GOD MODE monitoring overlay.",
        "icons": [
            {
                "src": "/icon.svg",
                "sizes": "512x512",
                "type": "image/svg+xml",
                "purpose": "any maskable",
            }
        ],
    }


def build_service_worker_script() -> str:
    """Return a service worker that caches only the static shell assets."""
    return """const CACHE_NAME = 'god-mode-shell-v1';
const SHELL_URLS = ['/', '/manifest.webmanifest', '/icon.svg'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_URLS))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/stream') || url.pathname === '/snapshot') return;

  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => caches.match('/'))
    );
    return;
  }

  if (SHELL_URLS.includes(url.pathname)) {
    event.respondWith(
      caches.match(event.request).then((cached) => cached || fetch(event.request))
    );
  }
});
"""


def build_icon_svg(title: str) -> str:
    """Return a simple vector icon for the installed PWA entry."""
    label = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    path_d = (
        "M120 120h72v16h-56v56h-16V120"
        "zm272 0v72h-16v-56h-56v-16h72"
        "zm-272 272h16v56h56v16h-72v-72"
        "zm272 0v72h-72v-16h56v-56h16z"
    )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" rx="48" fill="#000000"/>
  <rect
    x="48"
    y="48"
    width="416"
    height="416"
    rx="28"
    fill="#041b14"
    stroke="#00ffa0"
    stroke-width="12"
  />
  <path
    d="{path_d}"
    fill="#00ffa0"
    opacity="0.75"
  />
  <text
    x="256"
    y="236"
    text-anchor="middle"
    font-size="72"
    font-family="Courier New, monospace"
    fill="#00ffa0"
  >GM</text>
  <text
    x="256"
    y="292"
    text-anchor="middle"
    font-size="22"
    font-family="Noto Sans, sans-serif"
    fill="#9fe8c7"
  >{label}</text>
</svg>
"""
