/**
 * 파일 이름 : LogsApi.js
 * 기능 : 로그 조회, 삭제, 내보내기 등의 API 함수를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import { get, post, del } from './HttpClient';

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
    if (s === '') return;
    sp.set(k, s);
  });
  const qs = sp.toString();
  return qs ? `?${qs}` : '';
}

/**
 * 함수 이름 : fetchRecentLogs
 * 기능 : 최근 로그를 지정된 개수만큼 조회한다.
 * 매개변수 : limit - 조회할 로그 개수 (기본값: 50)
 * 반환값 : Promise - 로그 리스트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function fetchRecentLogs(limit = 50) {
  return get(`/logs/recent?limit=${Number(limit) || 50}`);
}

/**
 * 함수 이름 : fetchLogs
 * 기능 : 페이지네이션, 필터링, 정렬을 지원하는 로그 목록을 조회한다.
 * 매개변수 : params - 쿼리 파라미터 객체 (page, size, from, to, keyword, aiLabel, eventType, sort)
 * 반환값 : Promise - 페이지네이션된 로그 목록
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function fetchLogs(params = {}) {
  return get(`/logs${toQuery(params)}`);
}

/**
 * 함수 이름 : fetchLogDetail
 * 기능 : ID로 단일 로그의 상세 정보를 조회한다.
 * 매개변수 : id - 로그 ID
 * 반환값 : Promise - 로그 상세 정보
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function fetchLogDetail(id) {
  return get(`/logs/${id}`);
}

/**
 * 함수 이름 : deleteLog
 * 기능 : 단일 로그를 삭제한다.
 * 매개변수 : id - 삭제할 로그 ID
 * 반환값 : Promise - 삭제 결과
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function deleteLog(id) {
  return del(`/logs/${id}`);
}

/**
 * 함수 이름 : deleteLogs
 * 기능 : 여러 로그를 일괄 삭제한다.
 * 매개변수 : ids - 삭제할 로그 ID 리스트
 * 반환값 : Promise - 삭제 결과
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function deleteLogs(ids = []) {
  return post('/logs/delete', { ids });
}

/**
 * 함수 이름 : exportLogs
 * 기능 : 로그를 CSV 또는 JSON 형식으로 내보낸다.
 * 매개변수 : req - 내보내기 요청 객체 (format, filters 포함)
 * 반환값 : Promise - CSV 텍스트 또는 JSON 배열
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function exportLogs(req) {
  return post('/logs/export', req);
}
