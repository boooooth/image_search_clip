/**
 * Multimodal Fashion Search - Editorial Client-side Search
 */

document.addEventListener('DOMContentLoaded', () => {
    // Mode Tabs
    const tabText = document.getElementById('tab-text');
    const tabImage = document.getElementById('tab-image');
    const textSearchSection = document.getElementById('text-search-section');
    const imageSearchSection = document.getElementById('image-search-section');

    // Controls
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const topKSelect = document.getElementById('top-k-select');
    const vibeChips = document.querySelectorAll('.vibe-chip');
    const latencyVal = document.getElementById('latency-val');

    // Image Upload Controls
    const dropZone = document.getElementById('drop-zone');
    const imageFileInput = document.getElementById('image-file-input');
    const uploadPrompt = document.getElementById('upload-prompt');
    const previewWrapper = document.getElementById('preview-wrapper');
    const imagePreview = document.getElementById('image-preview');
    const previewFilename = document.getElementById('preview-filename');
    const removeImageBtn = document.getElementById('remove-image-btn');
    const searchImageBtn = document.getElementById('search-image-btn');

    // State Elements
    const initialState = document.getElementById('initial-state');
    const loadingState = document.getElementById('loading-state');
    const noResultsState = document.getElementById('no-results-state');
    const resultsGrid = document.getElementById('results-grid');

    let selectedFile = null;
    let debounceTimer = null;
    let currentMode = 'text'; // 'text' or 'image'

    // -------------------------------------------------------------------------
    // Mode Switching
    // -------------------------------------------------------------------------
    tabText.addEventListener('click', () => {
        currentMode = 'text';
        tabText.classList.add('active');
        tabImage.classList.remove('active');
        textSearchSection.classList.remove('hidden');
        imageSearchSection.classList.add('hidden');
        searchInput.focus();
    });

    tabImage.addEventListener('click', () => {
        currentMode = 'image';
        tabImage.classList.add('active');
        tabText.classList.remove('active');
        imageSearchSection.classList.remove('hidden');
        textSearchSection.classList.add('hidden');
    });

    // -------------------------------------------------------------------------
    // Perform Text Search
    // -------------------------------------------------------------------------
    async function performTextSearch(query) {
        const cleanQuery = query.trim();
        if (!cleanQuery) {
            showState('initial');
            latencyVal.textContent = 'LATENCY: 0.0 MS';
            return;
        }

        const topK = topKSelect.value || 10;
        showState('loading');

        try {
            const url = `/api/search?q=${encodeURIComponent(cleanQuery)}&top_k=${topK}`;
            const response = await fetch(url);

            if (!response.ok) {
                throw new Error(`Server status: ${response.status}`);
            }

            const data = await response.json();

            if (data.latency_ms !== undefined) {
                latencyVal.textContent = `LATENCY: ${data.latency_ms.toFixed(1)} MS`;
            }

            if (!data.results || data.results.length === 0) {
                showState('no-results');
            } else {
                renderResults(data.results);
                showState('results');
            }

        } catch (error) {
            console.error('Search error:', error);
            showState('no-results');
        }
    }

    // -------------------------------------------------------------------------
    // Perform Image Search
    // -------------------------------------------------------------------------
    async function performImageSearch() {
        if (!selectedFile) return;

        const topK = topKSelect.value || 10;
        showState('loading');

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('top_k', topK);

        try {
            const response = await fetch('/api/search-image', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`Server status: ${response.status}`);
            }

            const data = await response.json();

            if (data.latency_ms !== undefined) {
                latencyVal.textContent = `LATENCY: ${data.latency_ms.toFixed(1)} MS`;
            }

            if (!data.results || data.results.length === 0) {
                showState('no-results');
            } else {
                renderResults(data.results);
                showState('results');
            }

        } catch (error) {
            console.error('Image search error:', error);
            showState('no-results');
        }
    }

    // -------------------------------------------------------------------------
    // Render Results in Minimalist Editorial List
    // -------------------------------------------------------------------------
    function renderResults(items) {
        resultsGrid.innerHTML = '';

        items.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'result-item';

            const rankNum = index + 1;
            const rankFormatted = rankNum < 10 ? `0${rankNum}` : `${rankNum}`;

            // Image URL from local static server
            const imgSrc = `/images/${item.id}.jpg`;
            const scorePercent = (item.score * 100).toFixed(1);

            card.innerHTML = `
                <div class="item-thumb-wrapper">
                    <img class="item-thumb" src="${imgSrc}" alt="${escapeHtml(item.caption)}"
                         onerror="this.onerror=null; this.src='data:image/svg+xml,%3Csvg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'80\\' height=\\'100\\' viewBox=\\'0 0 80 100\\'%3E%3Crect width=\\'100%25\\' height=\\'100%25\\' fill=\\'%23f4f4f5\\'/ %3E%3Ctext x=\\'50%25\\' y=\\'50%25\\' dominant-baseline=\\'middle\\' text-anchor=\\'middle\\' fill=\\'%23a1a1aa\\' font-size=\\'10\\'%3ENo Photo%3C/text%3E%3C/svg%3E';">
                </div>
                <div class="item-content">
                    <div class="item-top">
                        <div class="item-title-group">
                            <span class="rank-tag">[${rankFormatted}]</span>
                            <span class="item-name" title="${escapeHtml(item.caption)}">${escapeHtml(item.caption)}</span>
                        </div>
                        <span class="score-tag">${item.score.toFixed(4)} (${scorePercent}%)</span>
                    </div>
                    <div class="item-meta-bar">
                        <span>ITEM ID: ${item.id}</span>
                        <span>•</span>
                        <span>CLIP WEIGHTED DUAL MATCH</span>
                    </div>
                </div>
            `;

            resultsGrid.appendChild(card);
        });
    }

    // -------------------------------------------------------------------------
    // State Manager
    // -------------------------------------------------------------------------
    function showState(state) {
        initialState.classList.add('hidden');
        loadingState.classList.add('hidden');
        noResultsState.classList.add('hidden');
        resultsGrid.classList.add('hidden');

        if (state === 'initial') {
            initialState.classList.remove('hidden');
        } else if (state === 'loading') {
            loadingState.classList.remove('hidden');
        } else if (state === 'no-results') {
            noResultsState.classList.remove('hidden');
        } else if (state === 'results') {
            resultsGrid.classList.remove('hidden');
        }
    }

    function escapeHtml(text) {
        if (!text) return '';
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return String(text).replace(/[&<>"']/g, m => map[m]);
    }

    // -------------------------------------------------------------------------
    // File Upload & Drag-and-Drop
    // -------------------------------------------------------------------------
    dropZone.addEventListener('click', (e) => {
        if (e.target !== removeImageBtn) {
            imageFileInput.click();
        }
    });

    imageFileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select a valid image file.');
            return;
        }

        selectedFile = file;
        previewFilename.textContent = file.name;

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            uploadPrompt.classList.add('hidden');
            previewWrapper.classList.remove('hidden');
            searchImageBtn.disabled = false;
        };
        reader.readAsDataURL(file);
    }

    removeImageBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        selectedFile = null;
        imageFileInput.value = '';
        imagePreview.src = '';
        uploadPrompt.classList.remove('hidden');
        previewWrapper.classList.add('hidden');
        searchImageBtn.disabled = true;
    });

    searchImageBtn.addEventListener('click', () => {
        performImageSearch();
    });

    // -------------------------------------------------------------------------
    // Event Listeners (Text Search)
    // -------------------------------------------------------------------------
    searchBtn.addEventListener('click', () => {
        performTextSearch(searchInput.value);
    });

    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            performTextSearch(searchInput.value);
        }
    });

    searchInput.addEventListener('input', () => {
        const val = searchInput.value;
        clearTimeout(debounceTimer);
        if (val.trim().length >= 3) {
            debounceTimer = setTimeout(() => {
                performTextSearch(val);
            }, 400);
        }
    });

    topKSelect.addEventListener('change', () => {
        if (currentMode === 'text' && searchInput.value.trim()) {
            performTextSearch(searchInput.value);
        } else if (currentMode === 'image' && selectedFile) {
            performImageSearch();
        }
    });

    vibeChips.forEach(chip => {
        chip.addEventListener('click', () => {
            const query = chip.getAttribute('data-query');
            searchInput.value = query;
            if (currentMode !== 'text') {
                tabText.click();
            }
            performTextSearch(query);
        });
    });
});
