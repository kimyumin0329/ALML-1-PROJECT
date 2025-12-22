/**
 * 파일 이름 : NotificationsApi.js
 * 기능 : 알림(윈도우 단위 AI 분석 결과) 조회 API 함수를 제공한다. 프론트엔드는 0-based 페이지를 사용하고 백엔드는 1-based를 사용하므로 변환을 수행한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import { get } from './HttpClient';

/**
 * 함수 이름 : toQuery
 * 기능 : 객체를 쿼리스트링으로 변환한다.
 * 매개변수 : params - 쿼리 파라미터 객체
 * 반환값 : string - 쿼리스트링
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
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

/**
 * 함수 이름 : normalizeLevel
 * 기능 : 위험도 레벨을 정규화한다. ALL이나 빈값이면 빈 문자열을 반환한다.
 * 매개변수 : level - 위험도 레벨 문자열
 * 반환값 : string - 정규화된 레벨 (DANGER|WARNING|SAFE 또는 빈 문자열)
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function normalizeLevel(level) {
  const v = String(level || '').trim().toUpperCase();
  if (v === 'DANGER' || v === 'WARNING' || v === 'SAFE') return v;
  return ''; // ALL/기타 => 빈값(=백엔드에서 전체)
}

/**
 * 함수 이름 : fetchAlerts
 * 기능 : 페이지네이션, 필터링, 정렬을 지원하는 알림 목록을 조회한다. 프론트엔드의 0-based 페이지를 백엔드의 1-based로 변환한다.
 * 매개변수 : params - 쿼리 파라미터 객체 (page: 0-based, size, keyword, level: ALL|DANGER|WARNING|SAFE, from: YYYY-MM-DD, to: YYYY-MM-DD, sort)
 * 반환값 : Promise - 페이지네이션된 알림 목록 (page는 0-based로 변환됨)
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export async function fetchAlerts({
  page = 0,
  size = 20,
  keyword = '',
  level = 'ALL',
  from = '',
  to = '',
  sort = 'createdAt,desc',
} = {}) {
  const lv = normalizeLevel(level);

  // ✅ 핵심: 백엔드 page는 1-based
  const backendPage = Math.max(0, Number(page) || 0) + 1;

  const res = await get(
    `/notifications${toQuery({
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

/**
 * 함수 이름 : fetchAlertDetail
 * 기능 : ID로 단일 알림의 상세 정보를 조회한다.
 * 매개변수 : id - 알림 ID
 * 반환값 : Promise - 알림 상세 정보 (windowStart, windowEnd, aiLabel, aiScore, topFamily, affectedFilesCount, affectedPaths 포함)
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function fetchAlertDetail(id) {
  return get(`/notifications/${id}`);
}

/**
 * 함수 이름 : fetchAlertStats
 * 기능 : 알림 통계를 일별 또는 주별로 조회한다. (현재는 사용하지 않음, 향후 구현 예정)
 * 매개변수 : params - 통계 파라미터 객체 (range: daily|weekly, from: YYYY-MM-DD, to: YYYY-MM-DD)
 * 반환값 : Promise - 통계 데이터
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export async function fetchAlertStats({
  range = 'daily',
  from = '',
  to = '',
} = {}) {
  // TODO: 향후 통계 API 구현 시 사용
  return {
    range: range || 'daily',
    from,
    to,
    series: [],
    counter: { total: 0, DANGER: 0, WARNING: 0, SAFE: 0, UNKNOWN: 0 },
  };
}
