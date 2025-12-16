// src/components/folders/FolderListManager.jsx
import React from 'react';

function FolderListManager({ folders, onAddFolder, onRemoveFolder }) {
  return (
    <div className="folder-panel">
      <div className="panel-header">
        <h2>감시 대상 폴더</h2>
      </div>

      <div className="folder-list">
        {folders.map((f) => (
          <div key={f.id} className="folder-item">
            <div className="folder-name">
              {f.name}
              <div style={{ fontSize: '11px', color: '#9ca3af', marginTop: '2px' }}>
                {f.path}
              </div>
            </div>
            <button className="btn-icon" onClick={() => onRemoveFolder?.(f.id)}>
              삭제
            </button>
          </div>
        ))}

        {folders.length === 0 && (
          <div style={{ fontSize: '13px', color: '#9ca3af' }}>
            아직 등록된 감시 폴더가 없습니다. 아래 버튼으로 추가해 주세요.
          </div>
        )}
      </div>

      <button className="btn btn-outline" onClick={onAddFolder}>
        폴더 추가
      </button>
    </div>
  );
}

export default FolderListManager;
