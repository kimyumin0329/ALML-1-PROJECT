/**
 * 파일 이름 : ProtectionStatusBadge.jsx
 * 기능 : 보호 상태 배지 컴포넌트. 현재 보호 상태와 최근 이벤트 시간을 표시한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';

/**
 * 함수 이름 : colorOf
 * 기능 : 상태 코드에 따른 색상을 반환한다.
 * 매개변수 : statusCode - 상태 코드 (DANGER, WARNING, SAFE)
 * 반환값 : string - 색상 코드
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function colorOf(statusCode) {
  if (statusCode === 'DANGER') return '#ef4444';
  if (statusCode === 'WARNING') return '#eab308';
  return '#22c55e';
}

/**
 * 함수 이름 : ProtectionStatusBadge
 * 기능 : 보호 상태 배지 컴포넌트. 상태 코드에 따라 색상이 변경되는 점과 상태 텍스트를 표시한다.
 * 매개변수 : protectionStatus - 보호 상태 텍스트, statusCode - 상태 코드, lastEventTime - 최근 이벤트 시간, summaryLoading - 로딩 상태, summaryError - 에러 객체
 * 반환값 : JSX.Element - 보호 상태 배지 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export default function ProtectionStatusBadge({
  protectionStatus = '안전',
  statusCode = 'SAFE',
  lastEventTime = 'N/A',
  summaryLoading = false,
  summaryError = null,
}) {
  return (
    <div className="status-panel">
      <div className="status-left">
        <div className="status-dot" style={{ backgroundColor: colorOf(statusCode) }} />
        <div>
          <div className="status-text">{protectionStatus}</div>
          <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 4 }}>
            최근 이벤트: {lastEventTime}
          </div>
          {summaryLoading && <div style={{ fontSize: 12, color: '#9ca3af' }}>요약 불러오는 중...</div>}
          {summaryError && <div style={{ fontSize: 12, color: '#ef4444' }}>요약 실패: {summaryError.message}</div>}
        </div>
      </div>
    </div>
  );
}
