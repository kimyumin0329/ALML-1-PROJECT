// src/api/NotificationsApi.js
import { get } from './HttpClient';

/**
 * Backend (정식):
 *  - GET  /alerts?page=1&size=50&from=&to=&level=&keyword=&sort=
 *  - GET  /alerts/{id}
 *  - GET  /alerts/stats?range=daily|weekly&from=&to=
 *
 * Front policy:
 *  - 프론트 page는 0-based 유지
 *  - 백엔드 page는 1-based이므로 요청 시 page+1, 응답은 page-1로 되돌림
 */

function toQuery(params = {}) {
  const sp = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v === undefined || v === null) return;
    const s = String(v).trim();
    if (!s) return;
    sp.set(k, s);
  });
  const qs = sp.toString();
  return qs ? `?${qs}` : '';
}

function normalizeLevel(level) {
  const v = String(level || '').trim().toUpperCase();
  if (v === 'DANGER' || v === 'WARNING' || v === 'SAFE') return v;
  return ''; // ALL/기타 => 빈값(=백엔드에서 전체)
}

/**
 * fetchAlerts
 * @param {Object} params
 * @param {number} params.page 0-based
 * @param {number} params.size
 * @param {string} params.keyword
 * @param {string} params.level ALL | DANGER | WARNING | SAFE
 * @param {string} params.from YYYY-MM-DD
 * @param {string} params.to YYYY-MM-DD
 * @param {string} params.sort collectedAt,desc 등
 */
export async function fetchAlerts({
  page = 0,
  size = 20,
  keyword = '',
  level = 'ALL',
  from = '',
  to = '',
  sort = 'collectedAt,desc',
} = {}) {
  const lv = normalizeLevel(level);

  // ✅ 핵심: 백엔드 page는 1-based
  const backendPage = Math.max(0, Number(page) || 0) + 1;

  const res = await get(
    `/alerts${toQuery({
      page: backendPage,
      size,
      keyword,
      level: lv, // ALL이면 빈값으로 넘어가서 전체
      from,
      to,
      sort,
    })}`
  );

  const items = Array.isArray(res?.items) ? res.items : [];
  const total = typeof res?.total === 'number' ? res.total : items.length;

  // ✅ 응답 page(1-based)를 다시 0-based로 복구
  const serverPage1 = typeof res?.page === 'number' ? res.page : backendPage;
  const clientPage0 = Math.max(0, serverPage1 - 1);

  return {
    items,
    total,
    page: clientPage0,
    size: typeof res?.size === 'number' ? res.size : size,
    mode: 'SERVER_PAGE',
  };
}

/** 알림 상세 */
export function fetchAlertDetail(id) {
  return get(`/alerts/${id}`);
}

/**
 * 알림 통계 (daily|weekly)
 * 백엔드: {range, from, to, series:[{date, warning, danger}]}
 *
 * ✅ 기존 프론트 UI(카운터형)를 유지하려고,
 *    series를 합산해서 counter도 같이 만들어서 반환해줌.
 */
export async function fetchAlertStats({
  range = 'daily', // daily | weekly
  from = '',
  to = '',
} = {}) {
  const rg = String(range || 'daily').trim().toLowerCase();
  const safeRange = rg === 'weekly' ? 'weekly' : 'daily';

  const res = await get(
    `/alerts/stats${toQuery({
      range: safeRange,
      from,
      to,
    })}`
  );

  const series = Array.isArray(res?.series) ? res.series : [];

  const counter = { total: 0, DANGER: 0, WARNING: 0, SAFE: 0, UNKNOWN: 0 };
  series.forEach((p) => {
    const w = Number(p?.warning || 0);
    const d = Number(p?.danger || 0);
    counter.WARNING += w;
    counter.DANGER += d;
  });
  counter.total = counter.WARNING + counter.DANGER;

  return {
    range: res?.range || safeRange,
    from: res?.from || from,
    to: res?.to || to,
    series,
    counter,
  };
}
