import React from 'react';

export default function NotificationDetailView({ notification }) {
  if (!notification) {
    return <p>알림 데이터가 없습니다.</p>;
  }

  return (
    <div className="notification-detail-card">
      <p>
        <strong>위험도:</strong> {notification.aiLabel || 'UNKNOWN'}
      </p>
      <p>
        <strong>이벤트 타입:</strong> {notification.eventType || '-'}
      </p>
      <p>
        <strong>수집 시각:</strong> {notification.collectedAt || '-'}
      </p>
      <p>
        <strong>파일 경로:</strong> {notification.path || '-'}
      </p>
      <p>
        <strong>파일 크기:</strong> {notification.size != null ? `${notification.size} bytes` : '-'}
      </p>
      <p>
        <strong>엔트로피:</strong> {notification.entropy != null ? notification.entropy : '-'}
      </p>
      <p>
        <strong>SHA-256 해시:</strong> {notification.hash || '-'}
      </p>
      <p>
        <strong>AI 점수:</strong> {notification.aiScore != null ? notification.aiScore : '-'}
      </p>

      <p>
        <strong>AI 상세 분석:</strong>
      </p>
      <pre className="notification-detail-text">
        {notification.aiDetail || '(상세 내용 없음)'}
      </pre>
    </div>
  );
}
