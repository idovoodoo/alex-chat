const CACHE_NAME = 'alex-chat-v1';
const urlsToCache = [
  '/static/index.html',
  '/static/images/alex-profile.webp'
];

// Install event - cache essential resources
self.addEventListener('install', (event) => {
  console.log('[ServiceWorker] Install');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[ServiceWorker] Caching app shell');
        return cache.addAll(urlsToCache);
      })
      .catch((err) => {
        console.error('[ServiceWorker] Cache failed:', err);
      })
  );
  self.skipWaiting();
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[ServiceWorker] Activate');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('[ServiceWorker] Removing old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    })
  );
  return self.clients.claim();
});

// Fetch event - serve from cache when offline, network first for API calls
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Network first for API calls
  if (url.pathname.startsWith('/chat') || 
      url.pathname.startsWith('/new_chat') || 
      url.pathname.startsWith('/db_status') ||
      url.pathname.startsWith('/debug')) {
    event.respondWith(
      fetch(request)
        .catch(() => {
          return new Response(
            JSON.stringify({ 
              error: 'You are offline. Please check your internet connection.' 
            }),
            { 
              headers: { 'Content-Type': 'application/json' },
              status: 503
            }
          );
        })
    );
    return;
  }

  // Cache first for static assets, then network
  event.respondWith(
    caches.match(request)
      .then((response) => {
        if (response) {
          return response;
        }
        return fetch(request)
          .then((response) => {
            // Don't cache if not a valid response
            if (!response || response.status !== 200 || response.type !== 'basic') {
              return response;
            }

            // Clone the response
            const responseToCache = response.clone();

            caches.open(CACHE_NAME)
              .then((cache) => {
                cache.put(request, responseToCache);
              });

            return response;
          })
          .catch((err) => {
            console.error('[ServiceWorker] Fetch failed:', err);
            // Return offline page or fallback
            return new Response('Offline - content not available', {
              status: 503,
              statusText: 'Service Unavailable'
            });
          });
      })
  );
});

// Handle background sync for offline messages (future enhancement)
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-messages') {
    console.log('[ServiceWorker] Background sync');
    event.waitUntil(
      // Could sync offline messages here
      Promise.resolve()
    );
  }
});

// Handle push notifications (future enhancement)
self.addEventListener('push', (event) => {
  console.log('[ServiceWorker] Push received');
  const data = event.data ? event.data.json() : {};
  
  const options = {
    body: data.body || 'New message from Alex',
    icon: '/static/images/alex-profile.webp',
    badge: '/static/images/alex-profile.webp',
    vibrate: [200, 100, 200],
    data: {
      url: data.url || '/static/index.html'
    }
  };

  event.waitUntil(
    self.registration.showNotification(data.title || 'Alex Chat', options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
  console.log('[ServiceWorker] Notification clicked');
  event.notification.close();

  event.waitUntil(
    clients.openWindow(event.notification.data.url || '/static/index.html')
  );
});
