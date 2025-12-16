import React, { memo, useMemo } from 'react';

function levelText(aiLabel) {
  if (!aiLabel) return 'UNKNOWN';
  return aiLabel;
}

function levelClass(aiLabel) {
  if (aiLabel === 'DANGER') return 'event-level-danger';
  if (aiLabel === 'WARNING') return 'event-level-warning';
  return 'event-level-info';
}

function NotificationItem({ item, onClick }) {
  const badge = useMemo(() => levelText(item?.aiLabel), [item?.aiLabel]);

  return (
    <li
      className="notification-item"
      onClick={() => onClick?.(item)}
      style={{ cursor: onClick ? 'pointer' : 'default' }}
    >
      <div className="notification-item-main">
        <span className="notification-title">
          [{badge}] {item?.eventType || '이벤트'}
        </span>
        <span className="notification-time">{item?.collectedAt || '-'}</span>
      </div>

      <div className="notification-item-sub">
        <span className="notification-path">{item?.path || '-'}</span>

        {item?.aiDetail && (
          <span className="notification-detail-preview">
            {item.aiDetail.length > 60 ? item.aiDetail.slice(0, 60) + '...' : item.aiDetail}
          </span>
        )}

        {/* 기존 CSS 이벤트 레벨 클래스 재사용(없으면 그냥 텍스트만 보임) */}
        <span className={'event-level ' + levelClass(item?.aiLabel)} style={{ marginLeft: 8 }}>
          {item?.aiLabel === 'DANGER' ? '위험' : item?.aiLabel === 'WARNING' ? '주의' : '정보'}
        </span>
      </div>
    </li>
  );
}

export default memo(NotificationItem);
