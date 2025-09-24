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

export function formatCurrency(value) {
    if (!Number.isFinite(value)) return '—';
    return `$${currencyFormatter.format(value)}`;
}

export function formatPercent(value) {
    if (!Number.isFinite(value)) return '—';
    return percentFormatter.format(value);
}

export function formatFactor(value) {
    if (!Number.isFinite(value)) return '—';
    return factorFormatter.format(value);
}

export function formatDecimal(value) {
    if (!Number.isFinite(value)) return '—';
    return decimalFormatter.format(value);
}

export function populateOriginSelectors(origins = [], selectedMin, selectedMax) {
    const minSelect = document.getElementById('min-origin');
    const maxSelect = document.getElementById('max-origin');
    if (!minSelect || !maxSelect) return;

    const buildOptions = (select, target) => {
        const previousValue = select.value;
        select.innerHTML = '';
        if (!origins.length) {
            const option = document.createElement('option');
            option.value = '';
            option.textContent = 'N/A';
            select.appendChild(option);
            select.disabled = true;
            return;
        }

        origins.forEach((origin) => {
            const option = document.createElement('option');
            option.value = origin;
            option.textContent = origin;
            select.appendChild(option);
        });

        const fallback = origins.includes(previousValue) ? previousValue : null;
        const resolved = origins.includes(target) ? target : fallback;
        select.value = resolved || (select === minSelect ? origins[0] : origins[origins.length - 1]);
        select.disabled = false;
    };

    buildOptions(minSelect, selectedMin);
    buildOptions(maxSelect, selectedMax);
}

export function updateSummaryCards(data) {
    const meanNode = document.getElementById('total-mean');
    const stdNode = document.getElementById('total-std');
    const cvNode = document.getElementById('coefficient-variation');
    const noteNode = document.getElementById('diagnostic-note');

    meanNode.textContent = formatCurrency(data.total_mean);
    stdNode.textContent = formatCurrency(data.total_std_error);
    cvNode.textContent = formatPercent(data.coefficient_of_variation);

    if (noteNode) {
        if (Number.isFinite(data.total_std_error) && data.total_std_error > 0) {
            const lower = Math.max(0, data.total_mean - 1.96 * data.total_std_error);
            const upper = data.total_mean + 1.96 * data.total_std_error;
            noteNode.textContent = `Approximate 95% normal interval: ${formatCurrency(lower)} – ${formatCurrency(upper)}.`;
        } else {
            noteNode.textContent = 'Approximate 95% normal interval: unavailable (zero standard error).';
        }
    }
}

function makeTrianglePlaceholder(body, columnCount) {
    const row = document.createElement('tr');
    row.innerHTML = `<td colspan="${columnCount}" class="text-center text-muted">Triangle unavailable.</td>`;
    body.appendChild(row);
}

export function renderTriangleTable(triangle, { onCellEdit } = {}) {
    const head = document.getElementById('triangle-table-head');
    const body = document.getElementById('triangle-table-body');
    if (!head || !body) return;
    head.innerHTML = '';
    body.innerHTML = '';

    if (!triangle || !Array.isArray(triangle.development_ages)) {
        makeTrianglePlaceholder(body, 4);
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
        makeTrianglePlaceholder(body, headerLabels.length);
        return;
    }

    triangle.rows.forEach((rowData) => {
        const tr = document.createElement('tr');
        const originCell = document.createElement('td');
        originCell.textContent = rowData.origin;
        tr.appendChild(originCell);

        (rowData.cells || []).forEach((cell) => {
            const td = document.createElement('td');
            td.dataset.origin = rowData.origin;
            td.dataset.age = cell.age;
            if (cell.status === 'observed') td.classList.add('cell-observed', 'cell-editable');
            if (cell.status === 'projected') td.classList.add('cell-projected');
            if (cell.status === 'empty') td.classList.add('cell-empty');
            td.textContent = formatCurrency(cell.value);
            td.classList.add('text-end');

            if (cell.status === 'observed' && typeof onCellEdit === 'function') {
                td.contentEditable = 'true';
                td.spellcheck = false;
                td.dataset.value = Number.isFinite(cell.value) ? String(cell.value) : '';
                td.addEventListener('focus', () => {
                    td.dataset.originalText = td.textContent;
                    td.classList.add('cell-active');
                    td.textContent = td.dataset.value || '';
                });
                td.addEventListener('blur', (event) => {
                    td.classList.remove('cell-active');
                    onCellEdit({
                        origin: rowData.origin,
                        age: cell.age,
                        text: event.target.textContent.trim(),
                        element: td,
                    });
                });
                td.addEventListener('keydown', (event) => {
                    if (event.key === 'Enter') {
                        event.preventDefault();
                        td.blur();
                    }
                });
            }

            tr.appendChild(td);
        });

        const ultimateCell = document.createElement('td');
        ultimateCell.classList.add('cell-observed', 'text-end');
        ultimateCell.textContent = formatCurrency(rowData.ultimate);
        tr.appendChild(ultimateCell);

        const stdCell = document.createElement('td');
        stdCell.classList.add('text-end');
        stdCell.textContent = formatCurrency(rowData.std_error);
        tr.appendChild(stdCell);

        body.appendChild(tr);
    });
}

export function renderFactorTable(rows, bodyId, formatter, columnCount) {
    const tbody = document.getElementById(bodyId);
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!rows || !rows.length) {
        const row = document.createElement('tr');
        row.innerHTML = `<td colspan="${columnCount}" class="text-center text-muted">Not available.</td>`;
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

export function updateDiagnosticsTable(diagnostics) {
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

export function updateDatasetBadge(datasetLabel) {
    const badge = document.getElementById('dataset-badge');
    if (badge) {
        badge.textContent = datasetLabel || '—';
    }
}

export function setLinearitySummaryText(content) {
    const summaryNode = document.getElementById('linearity-summary');
    if (summaryNode) {
        summaryNode.textContent = content;
    }
}

export function renderLinearitySelect(pairs, selectedLabel, onChange) {
    const select = document.getElementById('linearity-pair-select');
    if (!select) return;
    select.innerHTML = '';

    if (!pairs || !pairs.length) {
        const option = document.createElement('option');
        option.value = '';
        option.textContent = 'Insufficient data';
        select.appendChild(option);
        select.disabled = true;
        if (typeof onChange === 'function') onChange('');
        return;
    }

    select.disabled = false;
    pairs.forEach((pair) => {
        const option = document.createElement('option');
        option.value = pair.label;
        option.textContent = pair.label;
        select.appendChild(option);
    });
    const resolved = pairs.some((pair) => pair.label === selectedLabel)
        ? selectedLabel
        : pairs[0].label;
    select.value = resolved;
    if (typeof onChange === 'function') {
        select.onchange = (event) => onChange(event.target.value);
    }
}

export function renderLinearityPlot(pair) {
    const plotContainer = document.getElementById('linearity-plot');
    if (!plotContainer) return;

    if (!pair) {
        if (window.Plotly) {
            window.Plotly.purge(plotContainer);
        }
        setLinearitySummaryText('Linearity assessment unavailable for the selected development pair.');
        return;
    }

    const scatterTrace = {
        type: 'scatter',
        mode: 'markers',
        x: pair.points.map((point) => point.x),
        y: pair.points.map((point) => point.y),
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
        x: pair.line.map((point) => point.x),
        y: pair.line.map((point) => point.y),
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
        title: { text: pair.label, font: { size: 16 } },
        xaxis: { title: 'C_{ik}', tickformat: '$,~s' },
        yaxis: { title: 'C_{i,k+1}', tickformat: '$,~s' },
        legend: { orientation: 'h', yanchor: 'bottom', y: -0.2, xanchor: 'center', x: 0.5 },
    };

    if (window.Plotly) {
        window.Plotly.newPlot(plotContainer, [scatterTrace, lineTrace], layout, { responsive: true, displaylogo: false });
    }

    const pieces = [
        `Observations: ${pair.observations}`,
        `R²: ${formatDecimal(pair.r_squared)}`,
        `|Intercept| / Mean: ${formatPercent(pair.intercept_ratio)}`,
        `Mean residual %: ${formatPercent(pair.residual_mean_ratio)}`,
    ];
    setLinearitySummaryText(pieces.join(' · '));
}

export function updateEditableCell(cell, value) {
    if (!cell) return;
    cell.dataset.value = Number.isFinite(value) ? String(value) : '';
    cell.textContent = formatCurrency(value);
}

export function markCellError(cell, message) {
    if (!cell) return;
    cell.classList.add('cell-error');
    cell.title = message;
}

export function clearCellError(cell) {
    if (!cell) return;
    cell.classList.remove('cell-error');
    cell.removeAttribute('title');
}
