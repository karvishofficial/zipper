/**
 * Karvish Web State Distiller
 * Extracts actionable interactive elements from any webpage into Karvish Universal Candidates.
 * Coordinates are normalized to [0, 1000] density-independent grid space.
 */
(function() {
    try {
        const vw = Math.max(document.documentElement.clientWidth || 0, window.innerWidth || 0) || 1080;
        const vh = Math.max(document.documentElement.clientHeight || 0, window.innerHeight || 0) || 2400;

        function isVisible(el, rect) {
            if (!el || rect.width <= 0 || rect.height <= 0) return false;
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || style.visibility === 'hidden' || parseFloat(style.opacity) < 0.05) {
                return false;
            }
            // Check if within or reasonably near the viewport
            if (rect.bottom < -200 || rect.top > vh + 200 || rect.right < -200 || rect.left > vw + 200) {
                return false;
            }
            return true;
        }

        function getNormalizedBounds(rect) {
            const left = Math.max(0, Math.min(1000, Math.round((rect.left / vw) * 1000)));
            const top = Math.max(0, Math.min(1000, Math.round((rect.top / vh) * 1000)));
            const right = Math.max(0, Math.min(1000, Math.round((rect.right / vw) * 1000)));
            const bottom = Math.max(0, Math.min(1000, Math.round((rect.bottom / vh) * 1000)));
            return { left, top, right: Math.max(left + 1, right), bottom: Math.max(top + 1, bottom) };
        }

        function getRole(el, tag, type) {
            const ariaRole = (el.getAttribute('role') || '').toLowerCase();
            if (tag === 'INPUT') {
                if (type === 'search' || (el.name && el.name.toLowerCase().includes('search')) || (el.placeholder && el.placeholder.toLowerCase().includes('search'))) {
                    return 'SEARCH_FIELD';
                }
                if (type === 'button' || type === 'submit') return 'BUTTON';
                if (type === 'checkbox') return 'CHECKBOX';
                if (type === 'radio') return 'RADIO';
                return 'TEXT_FIELD';
            }
            if (tag === 'TEXTAREA') return 'TEXT_FIELD';
            if (tag === 'BUTTON' || ariaRole === 'button') return 'BUTTON';
            if (tag === 'A' || ariaRole === 'link') return 'LINK';
            if (tag === 'SELECT' || ariaRole === 'combobox' || ariaRole === 'listbox') return 'SELECT';
            if (/^H[1-6]$/.test(tag) || ariaRole === 'heading') return 'HEADER';
            if (ariaRole === 'tab') return 'TAB';
            if (ariaRole === 'checkbox') return 'CHECKBOX';
            if (ariaRole === 'radio') return 'RADIO';
            if (ariaRole === 'menuitem') return 'MENU_ITEM';
            if (tag === 'IMG' || ariaRole === 'img') return 'IMAGE';
            return 'LABEL';
        }

        function getLabel(el) {
            if (el.tagName === 'IMG') {
                const alt = el.getAttribute('alt');
                if (alt && alt.trim()) return alt.trim();
                const title = el.getAttribute('title');
                if (title && title.trim()) return title.trim();
                return 'Image';
            }
            if (el.tagName === 'SELECT') {
                const nameOrId = el.name || el.id || '';
                const selectedOpt = el.options && el.selectedIndex >= 0 ? el.options[el.selectedIndex] : null;
                const selText = selectedOpt ? (selectedOpt.text || selectedOpt.value || '').trim() : '';
                const opts = Array.from(el.options || []).map(o => (o.text || o.value || '').trim()).filter(Boolean);
                const optsSummary = opts.length > 0 ? (' [Options: ' + opts.slice(0, 31).join(',') + (opts.length > 31 ? '...' : '') + ']') : '';
                return (nameOrId ? (nameOrId + ' ') : '') + (selText ? 'Selected=' + selText : '') + optsSummary;
            }
            const ariaLabel = el.getAttribute('aria-label');
            if (ariaLabel && ariaLabel.trim()) return ariaLabel.trim();
            const placeholder = el.getAttribute('placeholder');
            if (placeholder && placeholder.trim()) return placeholder.trim();
            const value = el.value;
            if (value && typeof value === 'string' && value.trim()) return value.trim();
            const title = el.getAttribute('title');
            if (title && title.trim()) return title.trim();
            const alt = el.getAttribute('alt');
            if (alt && alt.trim()) return alt.trim();
            const text = el.innerText;
            if (text && text.trim()) return text.trim();
            if (el.tagName === 'INPUT') {
                const cls = (el.className || '').trim();
                const maxL = el.getAttribute('maxlength') || el.maxLength;
                if (type === 'password') {
                    return (cls ? cls + ' ' : '') + '(Password Field)';
                }
                if (maxL == 1 || /digit|otp|pin|code/i.test(cls)) {
                    return (cls ? cls + ' ' : '') + '(Digit Box)';
                }
                if (cls) return cls;
            }
            return '';
        }

        const selector = 'input, textarea, button, a, select, img, [role="button"], [role="link"], [role="tab"], [role="searchbox"], [role="combobox"], [role="listbox"], [role="option"], [role="menuitem"], [role="checkbox"], [role="radio"], [onclick], h1, h2, h3';
        const rawElements = Array.from(document.querySelectorAll(selector));

        // Also detect price and product summary nodes (e.g. ₹99, Rs. 150, $25)
        const textNodes = Array.from(document.querySelectorAll('span, div, p')).filter(el => {
            if (el.children.length > 0) return false;
            const t = el.innerText || '';
            return /(?:₹|Rs\.?|\$)\s*\d+(?:[.,]\d+)?/i.test(t) && t.length < 80;
        });

        const allCandidates = [];
        const seenElements = new Set();
        let index = 0;

        const processEl = (el, forceRole) => {
            if (!el || seenElements.has(el)) return;
            seenElements.add(el);

            const rect = el.getBoundingClientRect();
            if (!isVisible(el, rect)) return;

            const tag = el.tagName ? el.tagName.toUpperCase() : '';
            const type = (el.getAttribute('type') || '').toLowerCase();
            const ariaRole = (el.getAttribute('role') || '').toLowerCase();
            const role = forceRole || getRole(el, tag, type);
            const label = getLabel(el);

            // Filter out tiny icons or tracking pixels
            if (tag === 'IMG' && (rect.width < 32 || rect.height < 32)) {
                return;
            }

            // Skip empty non-input/non-button links
            if (!label && role !== 'TEXT_FIELD' && role !== 'SEARCH_FIELD' && role !== 'BUTTON' && role !== 'IMAGE') {
                return;
            }

            const bounds = getNormalizedBounds(rect);
            const karvishId = 'k_node_' + index;
            el.setAttribute('data-karvish-id', karvishId);

            const isClickable = role === 'BUTTON' || role === 'LINK' || role === 'TAB' || role === 'MENU_ITEM' || role === 'SELECT' || tag === 'BUTTON' || tag === 'A' || tag === 'SELECT' || el.hasAttribute('onclick');
            const isEditable = role === 'TEXT_FIELD' || role === 'SEARCH_FIELD';
            const isDisabled = !!(el.disabled || el.getAttribute('aria-disabled') === 'true');
            const isFocused = (document.activeElement === el);
            const isCheckable = (tag === 'INPUT' && (type === 'checkbox' || type === 'radio')) || ariaRole === 'checkbox' || ariaRole === 'radio';
            const isChecked = !!(el.checked || el.getAttribute('aria-checked') === 'true');
            const isSelected = !!(el.selected || el.getAttribute('aria-selected') === 'true');
            const isScrollable = (el.scrollHeight > el.clientHeight + 20) || (el.scrollWidth > el.clientWidth + 20);

            let href = null;
            if (tag === 'A' && el.href) {
                href = el.href;
            } else if (el.closest('a') && el.closest('a').href) {
                href = el.closest('a').href;
            } else if (el.getAttribute('href')) {
                try { href = new URL(el.getAttribute('href'), window.location.href).href; } catch(e) {}
            }
            const imgEl = tag === 'IMG' ? el : el.querySelector('img');
            const imageUrl = imgEl ? (imgEl.src || imgEl.getAttribute('data-src') || imgEl.getAttribute('data-lazy-src')) : null;

            // Spatial Region Classification
            let region = 'BODY';
            try {
                const style = window.getComputedStyle(el);
                const isFixed = style.position === 'fixed' || style.position === 'sticky';
                const zIndex = parseInt(style.zIndex, 10) || 0;
                if (zIndex >= 100 || (isFixed && rect.top > vh * 0.25 && rect.bottom < vh * 0.75)) {
                    region = 'MODAL';
                } else if (bounds.bottom <= 180) {
                    region = 'HEADER';
                } else if (bounds.top >= 820) {
                    region = 'FOOTER';
                }
            } catch (_) {}

            allCandidates.push({
                index: index,
                karvishId: karvishId,
                role: role,
                region: region,
                text: label.substring(0, 200),
                contentDescription: (el.getAttribute('aria-label') || '').substring(0, 120),
                resourceId: el.id || el.name || el.className || '',
                className: tag,
                hintText: el.getAttribute('placeholder') || null,
                isClickable: isClickable,
                isEditable: isEditable,
                isDisabled: isDisabled,
                isFocused: isFocused,
                isCheckable: isCheckable,
                isChecked: isChecked,
                isSelected: isSelected,
                isScrollable: isScrollable,
                bounds: [bounds.left, bounds.top, bounds.right, bounds.bottom],
                href: href || null,
                imageUrl: imageUrl || null
            });
            index++;
        };

        rawElements.forEach(el => processEl(el, null));
        textNodes.forEach(el => processEl(el, 'LIST_ITEM'));

        // Collect top visual images on the page (excluding site logos and small icons)
        const pageImages = Array.from(document.querySelectorAll('img'))
            .filter(img => {
                const r = img.getBoundingClientRect();
                const src = img.src || img.getAttribute('data-src') || '';
                const alt = img.getAttribute('alt') || '';
                const cls = (typeof img.className === 'string') ? img.className : '';
                const isLogo = /logo|favicon|nav_logo|googlelogo|icon_sprite/i.test(src) ||
                               /logo|favicon/i.test(alt) ||
                               /logo|favicon/i.test(cls);
                return isVisible(img, r) && r.width >= 50 && r.height >= 50 && !isLogo;
            })
            .map(img => img.src || img.getAttribute('data-src') || img.getAttribute('data-lazy-src') || '')
            .filter(src => src && (src.startsWith('http://') || src.startsWith('https://')) && !src.includes('data:image'))
            .slice(0, 5);

        return JSON.stringify({
            url: window.location.href,
            title: document.title,
            images: pageImages,
            candidates: allCandidates.slice(0, 100) // Keep top 100 elements for deep product coverage
        });
    } catch (err) {
        return JSON.stringify({
            url: window.location.href,
            title: document.title,
            error: err.toString(),
            candidates: []
        });
    }
})();
