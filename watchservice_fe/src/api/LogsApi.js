// src/api/LogsApi.js
import { get, post, del } from './HttpClient';

function toQuery(params = {}) {
  const sp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v === undefined || v === null) return;
    const s = String(v).trim();
    if (s === '') return;
    sp.set(k, s);
  });
  const qs = sp.toString();
  return qs ? `?${qs}` : '';
}

// ✅ 최근 로그
export function fetchRecentLogs(limit = 50) {
  return get(`/logs/recent?limit=${Number(limit) || 50}`);
}

// ✅ 명세형 전체 로그(있으면 사용, 없으면 훅에서 recent로 fallback)
export function fetchLogs(params = {}) {
  return get(`/logs${toQuery(params)}`);
}

// ✅ 단건 상세
export function fetchLogDetail(id) {
  return get(`/logs/${id}`);
}

// ✅ 단건 삭제
export function deleteLog(id) {
  return del(`/logs/${id}`);
}

// ✅ 선택 삭제
export function deleteLogs(ids = []) {
  return post('/logs/delete', { ids });
}

// ✅ 내보내기 (CSV면 text, JSON이면 array/object로 옴)
export function exportLogs(req) {
  return post('/logs/export', req);
}
