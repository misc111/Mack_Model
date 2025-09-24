import { fetchSummary, recalcSummary } from './api.js';
import {
    clearCellError,
    formatFactor,
    markCellError,
    populateOriginSelectors,
    renderFactorTable,
    renderLinearityPlot,
    renderLinearitySelect,
    renderTriangleTable,
    updateDatasetBadge,
    updateDiagnosticsTable,
    updateEditableCell,
    updateSummaryCards,
} from './ui.js';

const appState = {
    dataset: null,
    origins: [],
    minOrigin: null,
    maxOrigin: null,
    overrides: {},
    linearityPairs: [],
    selectedPairLabel: '',
    endpoints: {
        summary: '',
        recalc: '',
    },
    isLoading: false,
};

let requestToken = 0;

function setInterfaceBusy(isBusy) {
    appState.isLoading = isBusy;
    const form = document.getElementById('dataset-form');
    if (!form) return;
    form.classList.toggle('is-loading', isBusy);
    const submitButton = form.querySelector('button[type="submit"]');
    if (submitButton) submitButton.disabled = isBusy;
}

function parseNumericInput(text) {
    if (text === undefined || text === null) return null;
    const cleaned = text.replace(/[$,\s]/g, '');
    if (cleaned === '') return null;
    const value = Number.parseFloat(cleaned);
    return Number.isFinite(value) ? value : null;
}

function buildOverridesFromTriangle(triangle) {
    const overrides = {};
    if (!triangle || !Array.isArray(triangle.rows)) return overrides;

    triangle.rows.forEach((row) => {
        if (!row || !Array.isArray(row.cells)) return;
        const origin = row.origin;
        overrides[origin] = overrides[origin] || {};
        row.cells.forEach((cell) => {
            if (cell.status === 'observed' && Number.isFinite(cell.value)) {
                overrides[origin][cell.age] = Number(cell.value);
            }
        });
    });
    return overrides;
}

function updateLinearitySelection(pairs) {
    if (!pairs || !pairs.length) {
        appState.selectedPairLabel = '';
        renderLinearityPlot(null);
        return;
    }
    if (!pairs.some((pair) => pair.label === appState.selectedPairLabel)) {
        appState.selectedPairLabel = pairs[0].label;
    }
    renderLinearitySelect(pairs, appState.selectedPairLabel, (label) => {
        appState.selectedPairLabel = label;
        const nextPair = pairs.find((pair) => pair.label === label) || null;
        renderLinearityPlot(nextPair);
    });
    const currentPair = pairs.find((pair) => pair.label === appState.selectedPairLabel) || pairs[0];
    renderLinearityPlot(currentPair);
}

function applySummary(data) {
    if (!data) return;
    appState.dataset = data.dataset_key || appState.dataset;
    appState.origins = Array.isArray(data.origins) ? data.origins.slice() : [];
    appState.minOrigin = data.selected_min_origin || null;
    appState.maxOrigin = data.selected_max_origin || null;
    appState.overrides = buildOverridesFromTriangle(data.triangle_table);
    appState.linearityPairs = Array.isArray(data.linearity_pairs) ? data.linearity_pairs.slice() : [];

    populateOriginSelectors(appState.origins, appState.minOrigin, appState.maxOrigin);
    updateSummaryCards(data);
    renderTriangleTable(data.triangle_table, { onCellEdit: handleCellEdit });
    renderFactorTable(data.ldf_table, 'ldf-table-body', formatFactor, 3);
    renderFactorTable(data.cdf_table, 'cdf-table-body', formatFactor, 2);
    updateDiagnosticsTable(data.diagnostics);
    updateDatasetBadge(data.dataset_label);
    updateLinearitySelection(appState.linearityPairs);

    const datasetSelect = document.getElementById('dataset');
    if (datasetSelect && data.dataset_key) {
        datasetSelect.value = data.dataset_key;
    }
}

async function loadSummary({ dataset, minOrigin, maxOrigin }) {
    const requestId = ++requestToken;
    setInterfaceBusy(true);
    try {
        const data = await fetchSummary({
            endpoint: appState.endpoints.summary,
            dataset,
            minOrigin,
            maxOrigin,
        });
        if (requestId !== requestToken) return;
        applySummary(data);
    } catch (error) {
        alert(error.message);
    } finally {
        if (requestId === requestToken) {
            setInterfaceBusy(false);
        }
    }
}

async function recalculate() {
    const requestId = ++requestToken;
    setInterfaceBusy(true);
    try {
        const data = await recalcSummary({
            endpoint: appState.endpoints.recalc,
            dataset: appState.dataset,
            minOrigin: appState.minOrigin,
            maxOrigin: appState.maxOrigin,
            overrides: appState.overrides,
        });
        if (requestId !== requestToken) return;
        applySummary(data);
    } catch (error) {
        alert(error.message);
    } finally {
        if (requestId === requestToken) {
            setInterfaceBusy(false);
        }
    }
}

function handleCellEdit({ origin, age, text, element }) {
    const numericValue = parseNumericInput(text);
    if (numericValue === null) {
        markCellError(element, 'Please enter a numeric value.');
        const previous = appState.overrides?.[origin]?.[age];
        const fallback = Number.isFinite(previous) ? previous : 0;
        updateEditableCell(element, fallback);
        return;
    }

    clearCellError(element);
    updateEditableCell(element, numericValue);

    appState.overrides = appState.overrides || {};
    appState.overrides[origin] = appState.overrides[origin] || {};
    appState.overrides[origin][age] = numericValue;

    recalculate();
}

function handleFormSubmit(event) {
    event.preventDefault();
    if (appState.isLoading) return;
    const form = event.currentTarget;
    const datasetSelect = form.querySelector('#dataset');
    const minSelect = form.querySelector('#min-origin');
    const maxSelect = form.querySelector('#max-origin');

    const dataset = datasetSelect ? datasetSelect.value : appState.dataset;
    const minOrigin = minSelect ? minSelect.value : appState.minOrigin;
    const maxOrigin = maxSelect ? maxSelect.value : appState.maxOrigin;

    if (appState.origins.length) {
        const minIndex = appState.origins.indexOf(minOrigin);
        const maxIndex = appState.origins.indexOf(maxOrigin);
        if (minIndex > maxIndex) {
            alert('Oldest origin must be less than or equal to newest origin.');
            return;
        }
    }

    appState.dataset = dataset;
    appState.minOrigin = minOrigin;
    appState.maxOrigin = maxOrigin;
    loadSummary({ dataset, minOrigin, maxOrigin });
}

function initialise() {
    const body = document.body;
    appState.endpoints.summary = body.dataset.summaryEndpoint;
    appState.endpoints.recalc = body.dataset.recalcEndpoint;

    const form = document.getElementById('dataset-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }

    const datasetSelect = document.getElementById('dataset');
    const initialDataset = datasetSelect ? datasetSelect.value : undefined;
    appState.dataset = initialDataset;
    loadSummary({ dataset: initialDataset });
}

document.addEventListener('DOMContentLoaded', initialise);
