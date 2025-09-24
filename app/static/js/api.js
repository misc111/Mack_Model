async function handleResponse(response) {
    if (!response.ok) {
        let message = `Request failed with status ${response.status}`;
        try {
            const payload = await response.json();
            if (payload && payload.error) message = payload.error;
        } catch (error) {
            /* ignore JSON parse errors */
        }
        throw new Error(message);
    }
    return response.json();
}

export async function fetchSummary({ endpoint, dataset, minOrigin, maxOrigin }) {
    const params = new URLSearchParams();
    if (dataset) params.set('dataset', dataset);
    if (minOrigin) params.set('min_origin', minOrigin);
    if (maxOrigin) params.set('max_origin', maxOrigin);
    const query = params.toString();
    const url = query ? `${endpoint}?${query}` : endpoint;
    const response = await fetch(url, { method: 'GET' });
    return handleResponse(response);
}

export async function recalcSummary({ endpoint, dataset, minOrigin, maxOrigin, overrides }) {
    const payload = {
        dataset,
        min_origin: minOrigin,
        max_origin: maxOrigin,
        overrides,
    };
    const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    });
    return handleResponse(response);
}
