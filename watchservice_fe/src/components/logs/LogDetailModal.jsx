/**
 * 파일 이름 : LogDetailModal.jsx
 * 기능 : 로그 상세 정보를 모달로 표시하는 컴포넌트. 삭제 기능을 포함한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';

/**
 * 함수 이름 : LogDetailModal
 * 기능 : 로그 상세 정보를 모달로 표시한다.
 * 매개변수 : log - 로그 객체, onClose - 닫기 핸들러, onDelete - 삭제 핸들러
 * 반환값 : JSX.Element - 로그 상세 모달 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function LogDetailModal({ log, onClose, onDelete }) {
  if (!log) return null;

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2>로그 상세 (ID: {log.id})</h2>

        <div className="modal-content">
          <h3 style={{ marginTop: 0, marginBottom: 12 }}>기본 정보</h3>
          <p><strong>수집 시각:</strong> {log.collectedAt}</p>
          <p><strong>이벤트 타입:</strong> {log.eventType}</p>
          <p><strong>파일 경로:</strong> {log.path}</p>
          <p><strong>파일 존재 여부:</strong> {log.exists ? '존재' : '삭제됨'}</p>
          {log.existsBefore != null && (
            <p><strong>이전 존재 여부:</strong> {log.existsBefore ? '존재' : '없음'}</p>
          )}

          <h3 style={{ marginTop: 24, marginBottom: 12 }}>파일 변화 상세</h3>
          <p><strong>확장자 변화:</strong> {log.extBefore || '-'} → {log.extAfter || '-'}</p>
          <p><strong>파일 크기:</strong> {log.sizeBefore != null ? `${log.sizeBefore} → ${log.sizeAfter ?? log.size}` : log.size} bytes</p>
          <p><strong>크기 변화량:</strong> {log.sizeDiff != null ? `${log.sizeDiff >= 0 ? '+' : ''}${log.sizeDiff} bytes` : '-'}</p>
          <p><strong>엔트로피:</strong> {log.entropyBefore != null ? `${log.entropyBefore.toFixed(4)} → ${(log.entropyAfter ?? log.entropy ?? 0).toFixed(4)}` : (log.entropy != null ? log.entropy.toFixed(4) : '-')}</p>
          <p><strong>엔트로피 변화량:</strong> {log.entropyDiff != null ? `${log.entropyDiff >= 0 ? '+' : ''}${log.entropyDiff.toFixed(4)}` : '-'}</p>

          {log.hash && (
            <p><strong>SHA-256 해시:</strong> {log.hash}</p>
          )}
          {log.lastModifiedTime && (
            <p><strong>마지막 수정 시각:</strong> {log.lastModifiedTime}</p>
          )}

          {/*
          {(log.aiLabel || log.aiScore != null || log.aiDetail) && (
            <>
              <h3 style={{ marginTop: 24, marginBottom: 12 }}>AI 분석 결과 (참고)</h3>
              <p><strong>AI 판정:</strong> {log.aiLabel || '-'}</p>
              {log.aiScore != null && (
                <p><strong>AI 점수:</strong> {(log.aiScore * 100).toFixed(2)}%</p>
              )}
              {log.aiDetail && (
                <>
                  <p><strong>AI 상세:</strong></p>
                  <pre style={{ background: '#111827', color: '#e5e7eb', padding: 12, borderRadius: 8, maxHeight: 240, overflow: 'auto', fontSize: 12 }}>
                    {log.aiDetail}
                  </pre>
                </>
              )}
            </>
          )}
          */}
        </div>

        <div className="modal-actions" style={{ display: 'flex', gap: 8, justifyContent: 'flex-end' }}>
          <button className="btn" onClick={onDelete}>
            삭제
          </button>
          <button className="btn" onClick={onClose}>
            닫기
          </button>
        </div>
      </div>
    </div>
  );
}

export default LogDetailModal;
