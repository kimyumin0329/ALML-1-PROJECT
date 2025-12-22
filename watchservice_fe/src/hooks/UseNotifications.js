/**
 * 파일 이름 : UseNotifications.js
 * 기능 : 알림 조회를 위한 커스텀 훅. 페이지네이션, 필터링, 검색 기능을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { fetchAlerts, fetchAlertDetail, fetchAlertStats } from '../api/NotificationsApi';

/**
 * 함수 이름 : useNotifications
 * 기능 : 알림 조회를 위한 커스텀 훅. 페이지는 0-based로 통일되어 있다.
 * 매개변수 : initial - 초기 설정 객체 또는 페이지 크기 숫자 (기본값: 100)
 * 반환값 : Object - 알림 데이터, 페이지네이션, 필터, 로딩 상태 등을 포함한 객체
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function useNotifications(initial = 100) {
  const initialObj = useMemo(() => {
    if (typeof initial === 'number') {
      return { page: 0, size: Number(initial) || 100 };
    }
    return initial || {};
  }, [initial]);

  const [page, setPage] = useState(initialObj.page ?? 0);
  const [size, setSize] = useState(initialObj.size ?? 20);

  const [keyword, setKeyword] = useState(initialObj.keyword ?? '');
  const [level, setLevel] = useState(initialObj.level ?? 'ALL'); // ALL | DANGER | WARNING | SAFE
  const [from, setFrom] = useState(initialObj.from ?? '');
  const [to, setTo] = useState(initialObj.to ?? '');
  const [sort, setSort] = useState(initialObj.sort ?? 'createdAt,desc');

  const [items, setItems] = useState([]);
  const [total, setTotal] = useState(0);
  const [mode, setMode] = useState(''); // SERVER_PAGE | CLIENT_PAGE (디버깅용)

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const res = await fetchAlerts({
        page,
        size,
        keyword,
        level,
        from,
        to,
        sort,
      });

      setItems(res?.items || []);
      setTotal(typeof res?.total === 'number' ? res.total : 0);
      setMode(res?.mode || '');
    } catch (e) {
      setError(e);
      setItems([]);
      setTotal(0);
      setMode('');
    } finally {
      setLoading(false);
    }
  }, [page, size, keyword, level, from, to, sort]);

  useEffect(() => {
    load();
  }, [load]);

  // ✅ 기존 NotificationPage가 기대하는 limit/setLimit 호환
  const limit = size;
  const setLimit = (v) => {
    const next = Number(v) || 20;
    setSize(next);
    setPage(0);
  };

  const search = useCallback(() => {
    setPage(0);
  }, []);

  const refresh = useCallback(() => {
    load();
  }, [load]);

  return {
    notifications: items,
    total,
    mode,

    loading,
    error,

    // paging (0-based)
    page,
    setPage,
    size,
    setSize,

    // legacy alias
    limit,
    setLimit,

    // filters
    keyword,
    setKeyword,
    level,
    setLevel,
    from,
    setFrom,
    to,
    setTo,
    sort,
    setSort,

    search,
    refresh,

    fetchAlertDetail,
    fetchAlertStats,
  };
}
