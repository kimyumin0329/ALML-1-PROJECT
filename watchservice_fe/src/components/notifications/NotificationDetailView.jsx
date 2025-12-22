/**
 * 파일 이름 : NotificationDetailView.jsx
 * 기능 : 알림 상세 정보 뷰 컴포넌트. 알림의 모든 상세 정보를 표시한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';

/**
 * 함수 이름 : NotificationDetailView
 * 기능 : 알림 상세 정보 뷰 컴포넌트.
 * 매개변수 : notification - 알림 객체
 * 반환값 : JSX.Element - 알림 상세 정보 뷰 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
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
