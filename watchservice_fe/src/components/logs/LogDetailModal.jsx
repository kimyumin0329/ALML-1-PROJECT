// src/components/logs/LogDetailModal.jsx
import React from 'react';

function LogDetailModal({ log, onClose, onDelete }) {
  if (!log) return null;

  return (
    <div className="modal-backdrop">
      <div className="modal">
        <h2>로그 상세 (ID: {log.id})</h2>

        <div className="modal-content">
          <p><strong>수집 시각:</strong> {log.collectedAt}</p>
          <p><strong>이벤트 타입:</strong> {log.eventType}</p>
          <p><strong>파일 경로:</strong> {log.path}</p>
          <p><strong>파일 존재 여부:</strong> {log.exists ? '존재' : '삭제됨'}</p>
          <p><strong>파일 크기:</strong> {log.size} bytes</p>
          <p><strong>엔트로피:</strong> {log.entropy}</p>

          <p><strong>AI 판정:</strong> {log.aiLabel || '-'}</p>
          <p><strong>AI 점수:</strong> {log.aiScore ?? '-'}</p>

          <p><strong>AI 상세:</strong></p>
          <pre style={{ background: '#111827', color: '#e5e7eb', padding: 12, borderRadius: 8, maxHeight: 240, overflow: 'auto' }}>
            {log.aiDetail || '(상세 내용 없음)'}
          </pre>
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
