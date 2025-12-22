/**
 * 파일 이름 : NotificationList.jsx
 * 기능 : 알림 목록 컴포넌트. 알림 항목들을 리스트로 표시한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';
import NotificationItem from './NotificationItem';

/**
 * 함수 이름 : NotificationList
 * 기능 : 알림 목록 컴포넌트.
 * 매개변수 : items - 알림 배열, loading - 로딩 상태, error - 에러 객체, emptyText - 빈 목록 메시지, onItemClick - 항목 클릭 핸들러
 * 반환값 : JSX.Element - 알림 목록 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export default function NotificationList({
  items = [],
  loading = false,
  error = null,
  emptyText = '표시할 알림이 없습니다.',
  onItemClick,
}) {
  if (loading) return <p>불러오는 중...</p>;

  if (error) {
    return (
      <p style={{ color: 'red' }}>
        알림을 불러오는 중 오류가 발생했습니다: {error.message}
      </p>
    );
  }

  if (!items || items.length === 0) return <p>{emptyText}</p>;

  return (
    <ul className="notification-list">
      {items.map((item) => (
        <NotificationItem key={item.id} item={item} onClick={onItemClick} />
      ))}
    </ul>
  );
}
