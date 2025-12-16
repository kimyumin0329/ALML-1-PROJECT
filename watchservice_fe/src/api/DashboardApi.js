// src/api/DashboardApi.js

import { get } from './HttpClient';

/**
 * 대시보드 요약 정보 조회
 * GET /dashboard/summary
 */
export function fetchDashboardSummary() {
  return get('/dashboard/summary');
}
