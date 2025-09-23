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

let currentOrigins = [];
let currentLinearityPairs = [];

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

async function fetchSummary(dataset, minOrigin, maxOrigin) {
    const params = new URLSearchParams({ dataset });
    if (minOrigin) params.set('min_origin', minOrigin);
    if (maxOrigin) params.set('max_origin', maxOrigin);

    const response = await fetch(`/api/mack-distribution?${params.toString()}`);
    if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        const message = payload.error || `Request failed with status ${response.status}`;
        throw new Error(message);
    }
    return response.json();
}

function populateOriginSelectors(origins, selectedMin, selectedMax) {
    currentOrigins = Array.isArray(origins) ? origins.slice() : [];
    const minSelect = document.getElementById('min-origin');
    const maxSelect = document.getElementById('max-origin');
    if (!minSelect || !maxSelect) return;

    const buildOptions = (select, selected) => {
        const previousValue = select.value;
        select.innerHTML = '';
        if (!currentOrigins.length) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'N/A';
            select.appendChild(option);
            select.disabled = true;
            return;
        }
        currentOrigins.forEach((origin) => {
            const option = document.createElement('option');
            option.value = origin;
            option.textContent = origin;
            select.appendChild(option);
        });
        const target = selected && currentOrigins.includes(selected)
            ? selected
            : (currentOrigins.includes(previousValue) ? previousValue : null);
        select.value = target || (select === minSelect ? currentOrigins[0] : currentOrigins[currentOrigins.length - 1]);
        select.disabled = false;
    };

    buildOptions(minSelect, selectedMin);
    buildOptions(maxSelect, selectedMax);
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
        if (isFiniteNumber(data.total_std_error) && data.total_std_error > 0) {
            const lower = Math.max(0, data.total_mean - 1.96 * data.total_std_error);
            const upper = data.total_mean + 1.96 * data.total_std_error;
            noteNode.textContent = `Approximate 95% normal interval: ${formatCurrency(lower)} – ${formatCurrency(upper)}.`;
        } else {
            noteNode.textContent = 'Approximate 95% normal interval: unavailable (zero standard error).';
        }
    }
}

function renderTriangleTable(triangle) {
    const head = document.getElementById('triangle-table-head');
    const body = document.getElementById('triangle-table-body');
    if (!head || !body) return;
    head.innerHTML = '';
    body.innerHTML = '';

    if (!triangle || !Array.isArray(triangle.development_ages)) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="4" class="text-center text-muted">Triangle unavailable.</td>';
        body.appendChild(row);
        return;
    }

    const headerRow = document.createElement('tr');
    const headerLabels = ['Origin', ...triangle.development_ages, 'Ultimate', 'Std. Error'];
    headerLabels.forEach((label, index) => {
        const th = document.createElement('th');
        th.scope = 'col';
        th.textContent = label;
        if (index > 0) th.classList.add('text-end');
        headerRow.appendChild(th);
    });
    head.appendChild(headerRow);

    if (!triangle.rows || !triangle.rows.length) {
        const row = document.createElement('tr');
        row.innerHTML = `<td colspan="${headerLabels.length}" class="text-center text-muted">Triangle unavailable.</td>`;
        body.appendChild(row);
        return;
    }

    triangle.rows.forEach((rowData) => {
        const tr = document.createElement('tr');
        const originCell = document.createElement('td');
        originCell.textContent = rowData.origin;
        tr.appendChild(originCell);

        (rowData.cells || []).forEach((cell) => {
            const td = document.createElement('td');
            if (cell.status === 'observed') td.classList.add('cell-observed');
            if (cell.status === 'projected') td.classList.add('cell-projected');
            if (cell.status === 'empty') td.classList.add('cell-empty');
            td.textContent = formatCurrency(cell.value);
            tr.appendChild(td);
        });

        const ultimateCell = document.createElement('td');
        ultimateCell.classList.add('cell-observed');
        ultimateCell.textContent = formatCurrency(rowData.ultimate);
        tr.appendChild(ultimateCell);

        const stdCell = document.createElement('td');
        stdCell.textContent = formatCurrency(rowData.std_error);
        tr.appendChild(stdCell);

        body.appendChild(tr);
    });
}

function renderFactorTable(rows, bodyId, formatter, columnCount) {
    const tbody = document.getElementById(bodyId);
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!rows || !rows.length) {
        const row = document.createElement('tr');
        const colspan = columnCount || 1;
        row.innerHTML = `<td colspan="${colspan}" class="text-center text-muted">Not available.</td>`;
        tbody.appendChild(row);
        return;
    }

    rows.forEach((item) => {
        const tr = document.createElement('tr');
        if ('from_age' in item) {
            tr.innerHTML = `
                <td>${item.from_age}</td>
                <td>${item.to_age}</td>
                <td class="text-end">${formatter(item.factor)}</td>
            `;
        } else {
            tr.innerHTML = `
                <td>${item.age}</td>
                <td class="text-end">${formatter(item.factor)}</td>
            `;
        }
        tbody.appendChild(tr);
    });
}

function updateDiagnosticsTable(diagnostics) {
    const tbody = document.getElementById('diagnostics-table-body');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!diagnostics || !diagnostics.length) {
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

function updateDatasetBadge(datasetLabel) {
    const badge = document.getElementById('dataset-badge');
    if (badge) {
        badge.textContent = datasetLabel || '—';
    }
}

function populateLinearitySelect(pairs, selectedLabel) {
    const select = document.getElementById('linearity-pair-select');
    if (!select) return;
    select.innerHTML = '';

    if (!pairs || !pairs.length) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'Insufficient data';
        select.appendChild(option);
        select.disabled = true;
        return;
    }

    select.disabled = false;
    pairs.forEach((pair) => {
        const option = document.createElement('option');
        option.value = pair.label;
        option.textContent = pair.label;
        select.appendChild(option);
    });
    const targetLabel = selectedLabel && pairs.some((pair) => pair.label === selectedLabel)
        ? selectedLabel
        : pairs[0].label;
    select.value = targetLabel;
}

function updateLinearitySummary(pair) {
    const summaryNode = document.getElementById('linearity-summary');
    if (!summaryNode) return;
    if (!pair) {
        summaryNode.textContent = 'Linearity assessment unavailable for the selected development pair.';
        return;
    }
    const pieces = [
        `Observations: ${pair.observations}`,
        `R²: ${formatDecimal(pair.r_squared)}`,
        `|Intercept| / Mean: ${formatPercent(pair.intercept_ratio)}`,
        `Mean residual %: ${formatPercent(pair.residual_mean_ratio)}`,
    ];
    summaryNode.textContent = pieces.join(' · ');
}

function updateLinearityPlot(selectedLabel) {
    const plotContainer = document.getElementById('linearity-plot');
    if (!plotContainer) return;

    const targetPair = currentLinearityPairs.find((pair) => pair.label === selectedLabel);
    if (!targetPair) {
        Plotly.purge(plotContainer);
        updateLinearitySummary(null);
        return;
    }

    const scatterTrace = {
        type: 'scatter',
        mode: 'markers',
        x: targetPair.points.map((point) => point.x),
        y: targetPair.points.map((point) => point.y),
        marker: {
            color: '#6610f2',
            size: 8,
            opacity: 0.85,
        },
        name: 'Observed',
    };

    const lineTrace = {
        type: 'scatter',
        mode: 'lines',
        x: targetPair.line.map((point) => point.x),
        y: targetPair.line.map((point) => point.y),
        line: {
            color: '#0d6efd',
            width: 2,
        },
        name: 'VWA LDF Line',
    };

    const layout = {
        margin: { l: 60, r: 20, t: 40, b: 60 },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        title: { text: targetPair.label, font: { size: 16 } },
        xaxis: { title: 'C_{ik}', tickformat: '$,~s' },
        yaxis: { title: 'C_{i,k+1}', tickformat: '$,~s' },
        legend: { orientation: 'h', yanchor: 'bottom', y: -0.2, xanchor: 'center', x: 0.5 },
    };

    Plotly.newPlot(plotContainer, [scatterTrace, lineTrace], layout, { responsive: true, displaylogo: false });
    updateLinearitySummary(targetPair);
}

function handleFormSubmit(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const datasetSelect = form.querySelector('#dataset');
    const minSelect = form.querySelector('#min-origin');
    const maxSelect = form.querySelector('#max-origin');
    const submitButton = form.querySelector('button[type="submit"]');

    const dataset = datasetSelect.value;
    const minOrigin = minSelect.value;
    const maxOrigin = maxSelect.value;

    if (currentOrigins.length) {
        const minIndex = currentOrigins.indexOf(minOrigin);
        const maxIndex = currentOrigins.indexOf(maxOrigin);
        if (minIndex > maxIndex) {
            alert('Oldest origin must be less than or equal to newest origin.');
            return;
        }
    }

    try {
        form.classList.add('is-loading');
        submitButton.disabled = true;
        fetchSummary(dataset, minOrigin, maxOrigin)
            .then((data) => renderSummary(data))
            .catch((error) => alert(error.message))
            .finally(() => {
                form.classList.remove('is-loading');
                submitButton.disabled = false;
            });
    } catch (error) {
        alert(error.message);
        form.classList.remove('is-loading');
        submitButton.disabled = false;
    }
}

function renderSummary(data) {
    const datasetSelect = document.getElementById('dataset');
    if (datasetSelect && data.dataset) {
        datasetSelect.value = data.dataset;
    }
    populateOriginSelectors(data.origins, data.selected_min_origin, data.selected_max_origin);
    updateSummaryCards(data);
    renderTriangleTable(data.triangle_table);
    renderFactorTable(data.ldf_table, 'ldf-table-body', formatFactor, 3);
    renderFactorTable(data.cdf_table, 'cdf-table-body', formatFactor, 2);
    updateDiagnosticsTable(data.diagnostics);
    updateDatasetBadge(data.dataset_label);

    currentLinearityPairs = Array.isArray(data.linearity_pairs) ? data.linearity_pairs : [];
    const select = document.getElementById('linearity-pair-select');
    const defaultLabel = currentLinearityPairs.length ? currentLinearityPairs[0].label : '';
    populateLinearitySelect(currentLinearityPairs, defaultLabel);
    updateLinearityPlot(select ? select.value : defaultLabel);
}

function initialise() {
    const form = document.getElementById('dataset-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }

    const linearitySelect = document.getElementById('linearity-pair-select');
    if (linearitySelect) {
        linearitySelect.addEventListener('change', (event) => {
            updateLinearityPlot(event.target.value);
        });
    }

    const datasetSelect = document.getElementById('dataset');
    const initialDataset = datasetSelect ? datasetSelect.value : undefined;
    fetchSummary(initialDataset)
        .then((data) => renderSummary(data))
        .catch((error) => alert(error.message));
}

document.addEventListener('DOMContentLoaded', () => {
    initialise();
});
