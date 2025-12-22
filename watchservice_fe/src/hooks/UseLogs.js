/**
 * 파일 이름 : UseLogs.js
 * 기능 : 로그 조회를 위한 커스텀 훅. 페이지네이션, 필터링, 검색 기능을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import { useCallback, useEffect, useState } from 'react';
import { fetchLogs, fetchRecentLogs } from '../api/LogsApi';

/**
 * 함수 이름 : useLogs
 * 기능 : 로그 조회를 위한 커스텀 훅. 프론트엔드는 0-based 페이지를 사용하고 백엔드는 1-based를 사용하므로 변환을 수행한다.
 * 매개변수 : initialLimit - 초기 페이지 크기 (기본값: 50)
 * 반환값 : Object - 로그 데이터, 페이지네이션, 필터, 로딩 상태 등을 포함한 객체
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function useLogs(initialLimit = 50) {
  const [logs, setLogs] = useState([]);
  const [total, setTotal] = useState(0);

  // 0-based
  const [page, setPage] = useState(0);
  const [limit, setLimit] = useState(Number(initialLimit) || 50);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 서버 필터(실제 요청에 사용)
  const [filters, setFilters] = useState({
    keyword: '',
    from: '',
    to: '',
    sort: 'collectedAt,desc',
    aiLabel: '',     // SAFE/WARNING/DANGER 만, 나머지는 '' 로
    eventType: '',   // 필요하면 확장
  });

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      // ✅ 서버는 1-based page
      const serverPage = Number(page) + 1;

      const res = await fetchLogs({
        page: serverPage,
        size: limit,
        from: filters.from || '',
        to: filters.to || '',
        keyword: filters.keyword || '',
        aiLabel: filters.aiLabel || '',
        eventType: filters.eventType || '',
        sort: filters.sort || 'collectedAt,desc',
      });

      // 백엔드 응답: {items, page, size, total}
      const items = Array.isArray(res?.items) ? res.items : [];
      const t = typeof res?.total === 'number' ? res.total : items.length;

      setLogs(items);
      setTotal(t);

      // 서버가 1-based page를 내려주면 0-based로 동기화(선택)
      if (typeof res?.page === 'number') {
        const p0 = Math.max(0, res.page - 1);
        if (p0 !== page) setPage(p0);
      }
      if (typeof res?.size === 'number' && res.size !== limit) {
        setLimit(res.size);
      }
    } catch (e) {
      // fallback: recent 로그
      try {
        const recent = await fetchRecentLogs(limit);
        const items = Array.isArray(recent) ? recent : [];
        setLogs(items);
        setTotal(items.length);
      } catch (e2) {
        setLogs([]);
        setTotal(0);
        setError(e);
      }
    } finally {
      setLoading(false);
    }
  }, [page, limit, filters]);

  useEffect(() => {
    load();
  }, [load]);

  // 검색 버튼용: page만 0으로 리셋 (useEffect가 load)
  const search = useCallback(() => {
    setPage(0);
  }, []);

  const refresh = useCallback(() => {
    load();
  }, [load]);

  return {
    logs,
    loading,
    error,

    page,
    setPage,
    limit,
    setLimit,
    total,

    filters,
    setFilters,

    refresh,
    search,
  };
}
