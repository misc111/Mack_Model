const currencyFormatter = new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 0,
});
const factorFormatter = new Intl.NumberFormat('en-US', {
    minimumFractionDigits: 3,
    maximumFractionDigits: 3,
});
const percentFormatter = new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
});
const decimalFormatter = new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 2,
});

function isFiniteNumber(value) {
    return typeof value === 'number' && Number.isFinite(value);
}

function formatCurrency(value) {
    if (!isFiniteNumber(value)) return '—';
    return `$${currencyFormatter.format(value)}`;
}

function formatPercent(value) {
    if (!isFiniteNumber(value)) return '—';
    return percentFormatter.format(value);
}

function formatFactor(value) {
    if (!isFiniteNumber(value)) return '—';
    return factorFormatter.format(value);
}

function formatDecimal(value) {
    if (!isFiniteNumber(value)) return '—';
    return decimalFormatter.format(value);
}

async function fetchSummary(dataset) {
    const query = new URLSearchParams({ dataset });
    const response = await fetch(`/api/mack-distribution?${query.toString()}`);
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const message = payload.error || `Request failed with status ${response.status}`;
        throw new Error(message);
    }
    return response.json();
}

function updateSummaryCards(data) {
    const meanNode = document.getElementById('total-mean');
    const stdNode = document.getElementById('total-std');
    const cvNode = document.getElementById('coefficient-variation');
    const noteNode = document.getElementById('diagnostic-note');

    meanNode.textContent = formatCurrency(data.total_mean);
    stdNode.textContent = formatCurrency(data.total_std_error);
    cvNode.textContent = formatPercent(data.coefficient_of_variation);

    if (noteNode) {
        if (data.total_std_error > 0) {
            const lower = Math.max(0, data.total_mean - 1.96 * data.total_std_error);
            const upper = data.total_mean + 1.96 * data.total_std_error;
            noteNode.textContent = `Approximate 95% normal interval: ${formatCurrency(lower)} – ${formatCurrency(upper)}.`;
        } else {
            noteNode.textContent = 'Approximate 95% normal interval: not available (zero standard error).';
        }
    }
}

function updateDiagnosticsTable(diagnostics) {
    const tbody = document.getElementById('diagnostics-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!diagnostics || diagnostics.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="9" class="text-center text-muted">Diagnostics unavailable for this dataset.</td>';
        tbody.appendChild(row);
        return;
    }

    diagnostics.forEach((diag) => {
        const pair = `${diag.age} → ${diag.next_age}`;
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${pair}</td>
            <td>${diag.observations}</td>
            <td>${formatFactor(diag.development_factor)}</td>
            <td>${formatCurrency(diag.intercept)}</td>
            <td>${formatPercent(diag.intercept_ratio)}</td>
            <td>${formatDecimal(diag.r_squared)}</td>
            <td>${formatPercent(diag.residual_mean_ratio)}</td>
            <td>${formatDecimal(diag.variance_correlation)}</td>
            <td>${formatDecimal(diag.scaled_residual_std)}</td>
        `;
        tbody.appendChild(row);
    });
}

function updateOriginChart(ultimates, stdErrors, datasetLabel) {
    const origins = ultimates.map((row) => row.origin.split('T')[0]);
    const ultimateValues = ultimates.map((row) => row.ultimate);
    const stdLookup = new Map(stdErrors.map((row) => [row.origin, row.std_error]));
    const errorValues = origins.map((origin) => stdLookup.get(origin) || 0);

    const layout = {
        margin: { l: 60, r: 20, t: 40, b: 80 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        title: { text: `Origin Year Ultimates — ${datasetLabel}`, font: { size: 16 } },
        xaxis: { title: 'Origin', tickangle: -45 },
        yaxis: { title: 'Ultimate', tickformat: '$,~s' },
    };
    const trace = {
        type: 'bar',
        x: origins,
        y: ultimateValues,
        marker: { color: '#6610f2', opacity: 0.85 },
        error_y: {
            type: 'data',
            array: errorValues,
            visible: true,
            color: '#adb5bd',
            thickness: 1.5,
        },
    };
    Plotly.newPlot('origin-chart', [trace], layout, { responsive: true, displaylogo: false });
}

function updateOriginTable(ultimates, stdErrors) {
    const tbody = document.getElementById('origin-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';
    const stdLookup = new Map(stdErrors.map((row) => [row.origin, row.std_error]));
    ultimates.forEach((row) => {
        const stdError = stdLookup.get(row.origin) || 0;
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.origin.split('T')[0]}</td>
            <td>${formatCurrency(row.ultimate)}</td>
            <td>${formatCurrency(stdError)}</td>
        `;
        tbody.appendChild(tr);
    });
}

function updateDatasetBadge(datasetLabel) {
    const badge = document.getElementById('dataset-badge');
    if (badge) {
        badge.textContent = datasetLabel;
    }
}

async function handleFormSubmit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const datasetSelect = form.querySelector('#dataset');
    const submitButton = form.querySelector('button[type="submit"]');
    const dataset = datasetSelect.value;

    try {
        form.classList.add('is-loading');
        submitButton.disabled = true;
        const data = await fetchSummary(dataset);
        renderSummary(data);
    } catch (error) {
        alert(error.message);
    } finally {
        form.classList.remove('is-loading');
        submitButton.disabled = false;
    }
}

function renderSummary(data) {
    updateSummaryCards(data);
    updateDiagnosticsTable(data.diagnostics);
    updateOriginChart(data.origin_ultimates, data.origin_std_errors, data.dataset_label);
    updateOriginTable(data.origin_ultimates, data.origin_std_errors);
    updateDatasetBadge(data.dataset_label);
}

function initialise() {
    const form = document.getElementById('dataset-form');
    if (!form) return;
    form.addEventListener('submit', handleFormSubmit);

    const initialDataset = form.querySelector('#dataset').value;
    fetchSummary(initialDataset)
        .then((data) => renderSummary(data))
        .catch((error) => alert(error.message));
}

document.addEventListener('DOMContentLoaded', () => {
    initialise();
});
