// src/pages/settings/SettingFoldersPage.jsx
import React from 'react';
import { useWatchedFolders } from '../../hooks/UseWatchedFolders';
import FolderListManager from '../../components/folders/FolderListManager';

function SettingFoldersPage() {
  const {
    folders,
    loading,
    error,
    refresh,
    promptAndAddFolder,
    removeFolder,
  } = useWatchedFolders();

  return (
    <div className="page-container">
      <h1>감시 폴더 설정</h1>
      <p style={{ fontSize: 14, color: '#9ca3af', marginBottom: 16 }}>
        랜섬웨어 감시 대상이 될 폴더를 등록/삭제할 수 있습니다.
        메인 보드에서 사용하는 폴더 목록과 동일하게 동기화됩니다.
      </p>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <button className="btn" onClick={refresh} disabled={loading}>
          새로고침
        </button>
        {loading && <span style={{ fontSize: 13, color: '#9ca3af' }}>불러오는 중...</span>}
        {error && (
          <span style={{ fontSize: 13, color: 'red' }}>
            서버 연동 오류: {error.message}
          </span>
        )}
      </div>

      <FolderListManager
        folders={folders || []}
        onAddFolder={promptAndAddFolder}
        onRemoveFolder={removeFolder}
      />
    </div>
  );
}

export default SettingFoldersPage;
