/**
 * 파일 이름 : RecentEventsPanel.jsx
 * 기능 : 최근 탐지 이벤트 패널 컴포넌트. 최근 이벤트 목록을 표시한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';

/**
 * 함수 이름 : levelOf
 * 기능 : 로그의 AI 라벨에 따라 위험도 레벨을 반환한다.
 * 매개변수 : log - 로그 객체
 * 반환값 : string - 위험도 레벨 (danger, warning, info)
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function levelOf(log) {
  if (log?.aiLabel === 'DANGER') return 'danger';
  if (log?.aiLabel === 'WARNING') return 'warning';
  return 'info';
}

/**
 * 함수 이름 : RecentEventsPanel
 * 기능 : 최근 탐지 이벤트 패널 컴포넌트.
 * 매개변수 : events - 이벤트 배열, loading - 로딩 상태, error - 에러 객체, onEventClick - 이벤트 클릭 핸들러
 * 반환값 : JSX.Element - 최근 이벤트 패널 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export default function RecentEventsPanel({
  events = [],
  loading = false,
  error = null,
  onEventClick,
}) {
  return (
    <div className="recent-events-panel">
      <div className="panel-header">
        <h2>최근 탐지 이벤트</h2>
      </div>

      {loading && <p>최근 이벤트를 불러오는 중...</p>}
      {error && <p style={{ color: 'red' }}>최근 이벤트 로드 실패: {error.message}</p>}

      {!loading && !error && (!events || events.length === 0) && (
        <p style={{ color: '#9ca3af' }}>표시할 이벤트가 없습니다.</p>
      )}

      <div className="recent-events-list">
        {(events || []).map((ev) => {
          const lv = levelOf(ev);
          return (
            <div
              key={ev.id}
              className="recent-event-item"
              role="button"
              tabIndex={0}
              onClick={() => onEventClick?.(ev)}
              onKeyDown={(e) => (e.key === 'Enter' ? onEventClick?.(ev) : null)}
              style={{ cursor: 'pointer' }}
            >
              <div className="recent-event-main">
                <div style={{ fontSize: 12, color: '#9ca3af' }}>{ev.collectedAt}</div>
                <div style={{ fontWeight: 600, marginTop: 2 }}>
                  {ev.eventType} · {ev.path}
                </div>
              </div>

              <span
                className={
                  'event-level ' +
                  (lv === 'danger'
                    ? 'event-level-danger'
                    : lv === 'warning'
                    ? 'event-level-warning'
                    : 'event-level-info')
                }
              >
                {lv === 'danger' ? '위험' : lv === 'warning' ? '주의' : '정보'}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
