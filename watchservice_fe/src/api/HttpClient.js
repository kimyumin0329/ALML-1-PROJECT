// src/api/HttpClient.js

const BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080';

async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;

  const finalOptions = {
    ...options,
    headers: {
      ...(options.headers || {}),
    },
  };

  const res = await fetch(url, finalOptions);

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} ${res.statusText}${text ? ` - ${text}` : ''}`);
  }

  const ct = (res.headers.get('content-type') || '').toLowerCase();
  if (ct.includes('application/json')) return res.json();
  return res.text();
}

export function get(path) {
  return request(path, { method: 'GET' });
}

export function del(path) {
  return request(path, { method: 'DELETE' });
}

export function post(path, body) {
  // body 없으면 쿼리스트링만 사용하는 POST 같은 케이스 대응
  if (body === undefined) return request(path, { method: 'POST' });

  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export function put(path, body) {
  return request(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
}
