/**
 * 파일 이름 : NotificationStatsPage.jsx
 * 기능 : 알림 통계 페이지. 알림 통계를 차트로 표시한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React, { useEffect, useMemo, useState } from 'react';
import { fetchAlertStats } from '../../api/NotificationsApi';
import NotificationStatusChart from '../../components/notifications/NotificationStatusChart';

/**
 * 함수 이름 : NotificationStatsPage
 * 기능 : 알림 통계 페이지 컴포넌트.
 * 매개변수 : 없음
 * 반환값 : JSX.Element - 알림 통계 페이지 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function NotificationStatsPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // 간단 통계 (카운터)
  const [counter, setCounter] = useState({ total: 0, DANGER: 0, WARNING: 0, SAFE: 0, UNKNOWN: 0 });

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchAlertStats({ level: 'ALL' });
      setCounter(res?.counter || { total: 0, DANGER: 0, WARNING: 0, SAFE: 0, UNKNOWN: 0 });
    } catch (e) {
      setError(e);
      setCounter({ total: 0, DANGER: 0, WARNING: 0, SAFE: 0, UNKNOWN: 0 });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const stats = useMemo(() => counter, [counter]);

  return (
    <div className="page-container">
      <h1>알림 통계</h1>

      <button className="btn" onClick={load} style={{ marginBottom: 12 }} disabled={loading}>
        새로고침
      </button>

      {loading && <p>불러오는 중...</p>}
      {error && <p style={{ color: 'red' }}>통계를 불러오는 중 오류: {error.message}</p>}

      {!loading && !error && <NotificationStatusChart stats={stats} />}
    </div>
  );
}

export default NotificationStatsPage;
