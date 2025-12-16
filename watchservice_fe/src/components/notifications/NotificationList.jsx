// src/components/notifications/NotificationList.jsx
import React from 'react';
import NotificationItem from './NotificationItem';

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
