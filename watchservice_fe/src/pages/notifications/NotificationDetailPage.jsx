// src/pages/notifications/NotificationDetailPage.jsx
import React, { useEffect, useState } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { fetchAlertDetail } from '../../api/NotificationsApi';

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
            <p><strong>위험도:</strong> {notification.aiLabel || 'UNKNOWN'}</p>
            <p><strong>이벤트 타입:</strong> {notification.eventType}</p>
            <p><strong>수집 시각:</strong> {notification.collectedAt}</p>
            <p><strong>파일 경로:</strong> {notification.path}</p>
            <p><strong>파일 크기:</strong> {notification.size} bytes</p>
            <p><strong>엔트로피:</strong> {notification.entropy != null ? notification.entropy : '-'}</p>
            <p><strong>SHA-256 해시:</strong> {notification.hash || '-'}</p>
            <p><strong>AI 점수:</strong> {notification.aiScore != null ? notification.aiScore : '-'}</p>

            <p><strong>AI 상세 분석:</strong></p>
            <pre className="notification-detail-text">
              {notification.aiDetail || '(상세 내용 없음)'}
            </pre>
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
