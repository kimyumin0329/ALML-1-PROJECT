/**
 * 파일 이름 : DashboardApi.js
 * 기능 : 대시보드 요약 정보 조회 API 함수를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import { get } from './HttpClient';

/**
 * 함수 이름 : fetchDashboardSummary
 * 기능 : 대시보드 요약 정보를 조회한다. 최근 알림, 위험도 상태, 감시 폴더 경로 등을 포함한다.
 * 매개변수 : 없음
 * 반환값 : Promise - 대시보드 요약 정보
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function fetchDashboardSummary() {
  return get('/dashboard/summary');
}
