/**
 * 파일 이름 : ScanControlPanel.jsx
 * 기능 : 스캔 제어 패널 컴포넌트. 즉시 검사, 감시 시작/중지, 일시 중지 버튼과 진행률 바를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';

/**
 * 함수 이름 : ScanControlPanel
 * 기능 : 스캔 제어 패널 컴포넌트.
 * 매개변수 : isWatching - 감시 중 여부, isScanning - 검사 중 여부, scanProgress - 검사 진행률 (0-100), onToggleWatch - 감시 토글 핸들러, onScanNow - 즉시 검사 핸들러, onPause - 일시 중지 핸들러
 * 반환값 : JSX.Element - 스캔 제어 패널 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
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
