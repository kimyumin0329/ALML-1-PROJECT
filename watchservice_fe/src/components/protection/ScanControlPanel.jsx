// src/components/protection/ScanControlPanel.jsx
import React from 'react';

export default function ScanControlPanel({
  isWatching,
  isScanning,
  scanProgress,
  onToggleWatch,
  onScanNow,
  onPause,
}) {
  const safeProgress = Math.max(0, Math.min(100, Number(scanProgress || 0)));

  return (
    <div className="scan-panel">
      <div className="scan-controls">
        <button className="btn" onClick={onScanNow} disabled={isScanning}>
          즉시 검사
        </button>

        <button className="btn" onClick={onToggleWatch}>
          {isWatching ? '감시 중지' : '감시 시작'}
        </button>

        <button className="btn" onClick={onPause} disabled={!isScanning}>
          일시 중지
        </button>
      </div>

      <div className="scan-progress">
        <div className="scan-progress-bar">
          <div className="scan-progress-fill" style={{ width: `${safeProgress}%` }} />
        </div>
        <div className="scan-progress-text">진행률: {safeProgress}%</div>
      </div>
    </div>
  );
}
