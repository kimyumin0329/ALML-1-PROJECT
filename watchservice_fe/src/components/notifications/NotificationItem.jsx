/**
 * 파일 이름 : NotificationItem.jsx
 * 기능 : 알림 항목 컴포넌트. 개별 알림의 정보를 표시한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React, { memo, useMemo } from 'react';

/**
 * 함수 이름 : levelText
 * 기능 : AI 라벨을 텍스트로 변환한다.
 * 매개변수 : aiLabel - AI 라벨
 * 반환값 : string - 라벨 텍스트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function levelText(aiLabel) {
  if (!aiLabel) return 'UNKNOWN';
  return aiLabel;
}

/**
 * 함수 이름 : levelClass
 * 기능 : AI 라벨에 따른 CSS 클래스명을 반환한다.
 * 매개변수 : aiLabel - AI 라벨
 * 반환값 : string - CSS 클래스명
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function levelClass(aiLabel) {
  if (aiLabel === 'DANGER') return 'event-level-danger';
  if (aiLabel === 'WARNING') return 'event-level-warning';
  return 'event-level-info';
}

/**
 * 함수 이름 : NotificationItem
 * 기능 : 알림 항목 컴포넌트.
 * 매개변수 : item - 알림 객체, onClick - 클릭 핸들러
 * 반환값 : JSX.Element - 알림 항목 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
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
