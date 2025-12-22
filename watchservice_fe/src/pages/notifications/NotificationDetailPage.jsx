/**
 * 파일 이름 : NotificationDetailPage.jsx
 * 기능 : 알림 상세 페이지. 알림의 상세 정보를 표시한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { fetchAlertDetail } from '../../api/NotificationsApi';

/**
 * 함수 이름 : NotificationDetailPage
 * 기능 : 알림 상세 페이지 컴포넌트.
 * 매개변수 : 없음
 * 반환값 : JSX.Element - 알림 상세 페이지 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function NotificationDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  const stateItem = location.state?.notification;

  const [notification, setNotification] = useState(stateItem || null);
  const [loading, setLoading] = useState(!stateItem);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (stateItem) return;

    let mounted = true;
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const data = await fetchAlertDetail(id);
        if (mounted) setNotification(data);
      } catch (e) {
        if (mounted) setError(e);
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, [id, stateItem]);

  return (
    <div className="page-container">
      <h1>알림 상세 (ID: {id})</h1>

      {loading && <p>불러오는 중...</p>}
      {error && <p style={{ color: 'red' }}>상세 로드 오류: {error.message}</p>}

      {!loading && !error && !notification && (
        <>
          <p>알림 데이터를 찾을 수 없습니다.</p>
          <button className="btn" onClick={() => navigate('/notifications')}>
            알림 목록으로 돌아가기
          </button>
        </>
      )}

      {!loading && !error && notification && (
        <>
          <div className="notification-detail-card">
            <h2>윈도우 정보</h2>
            <p><strong>윈도우 시작:</strong> {notification.windowStart}</p>
            <p><strong>윈도우 종료:</strong> {notification.windowEnd}</p>
            <p><strong>생성 시각:</strong> {notification.createdAt}</p>

            <h2 style={{ marginTop: 24 }}>AI 분석 결과</h2>
            <p><strong>위험도:</strong> {notification.aiLabel || 'UNKNOWN'}</p>
            <p><strong>AI 점수:</strong> {notification.aiScore != null ? (notification.aiScore * 100).toFixed(2) + '%' : '-'}</p>
            <p><strong>탑 패밀리:</strong> {notification.topFamily || '-'}</p>
            {notification.aiDetail && (
              <>
                <p><strong>AI 상세:</strong></p>
                <pre className="notification-detail-text">
                  {notification.aiDetail}
                </pre>
              </>
            )}

            <h2 style={{ marginTop: 24 }}>영향 받은 파일</h2>
            <p><strong>영향 받은 파일 개수:</strong> {notification.affectedFilesCount || 0}개</p>
            {notification.affectedPaths && notification.affectedPaths.length > 0 && (
              <>
                <p><strong>파일 경로 목록:</strong></p>
                <ul style={{ background: '#111827', padding: 16, borderRadius: 8, maxHeight: 400, overflow: 'auto' }}>
                  {notification.affectedPaths.map((path, idx) => (
                    <li key={idx} style={{ marginBottom: 8, fontFamily: 'monospace', fontSize: 13 }}>
                      {path}
                    </li>
                  ))}
                </ul>
              </>
            )}
          </div>

          <button className="btn" onClick={() => navigate('/notifications')}>
            알림 목록으로 돌아가기
          </button>
        </>
      )}
    </div>
  );
}

export default NotificationDetailPage;
