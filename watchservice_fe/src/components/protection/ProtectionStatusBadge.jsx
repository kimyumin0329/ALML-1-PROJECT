// src/components/protection/ProtectionStatusBadge.jsx
import React from 'react';

function colorOf(statusCode) {
  if (statusCode === 'DANGER') return '#ef4444';
  if (statusCode === 'WARNING') return '#eab308';
  return '#22c55e';
}

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
