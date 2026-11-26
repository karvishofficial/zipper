/**
 * Karvish Adaptive Network-Idle & DOM Stability Observer.
 * Tracks in-flight HTTP requests (fetch + XHR) and DOM mutations
 * to dynamically adapt execution speed to 5G vs slow 3G networks.
 */
(function() {
    if (window.__karvishNetworkObserverInstalled) return;
    window.__karvishNetworkObserverInstalled = true;
    window.__karvishNetworkState = {
        inflightRequests: 0,
        lastMutationTime: Date.now(),
        lastRequestEndTime: Date.now()
    };

    // 1. Intercept fetch
    const origFetch = window.fetch;
    if (origFetch) {
        window.fetch = function(...args) {
            window.__karvishNetworkState.inflightRequests++;
            return origFetch.apply(this, args).finally(() => {
                window.__karvishNetworkState.inflightRequests = Math.max(0, window.__karvishNetworkState.inflightRequests - 1);
                window.__karvishNetworkState.lastRequestEndTime = Date.now();
            });
        };
    }

    // 2. Intercept XMLHttpRequest
    const origOpen = XMLHttpRequest.prototype.open;
    const origSend = XMLHttpRequest.prototype.send;
    XMLHttpRequest.prototype.open = function(...args) {
        this.__karvishTracked = true;
        return origOpen.apply(this, args);
    };
    XMLHttpRequest.prototype.send = function(...args) {
        if (this.__karvishTracked) {
            window.__karvishNetworkState.inflightRequests++;
            const cleanup = () => {
                if (this.__karvishCleaned) return;
                this.__karvishCleaned = true;
                window.__karvishNetworkState.inflightRequests = Math.max(0, window.__karvishNetworkState.inflightRequests - 1);
                window.__karvishNetworkState.lastRequestEndTime = Date.now();
            };
            this.addEventListener('loadend', cleanup);
            this.addEventListener('error', cleanup);
            this.addEventListener('abort', cleanup);
        }
        return origSend.apply(this, args);
    };

    // 3. Monitor DOM mutations
    try {
        const observer = new MutationObserver(() => {
            window.__karvishNetworkState.lastMutationTime = Date.now();
        });
        observer.observe(document.documentElement, {
            childList: true,
            subtree: true,
            attributes: false
        });
    } catch (_) {}
})();
