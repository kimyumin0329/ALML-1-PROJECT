// src/hooks/UseWatchedFolders.js
import { useCallback, useEffect, useState } from 'react';
import {
  fetchWatchedFolders,
  pickFolderPath,
  createWatchedFolder,
  deleteWatchedFolder,
} from '../api/SettingApi';

function guessNameFromPath(path) {
  if (!path) return '폴더';
  const parts = String(path).split(/[/\\]+/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : '폴더';
}

export function useWatchedFolders() {
  const [folders, setFolders] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchWatchedFolders();
      const list = Array.isArray(data) ? data : (data?.items ?? []);

      setFolders(
        list.map((it) => ({
          id: it.id ?? it.folderId ?? it.path,
          name: it.name ?? it.folderName ?? guessNameFromPath(it.path),
          path: it.path,
        }))
      );
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]); // ✅ eslint 경고 해결 포인트

  // ✅ 기존 페이지가 onAddFolder로 쓰는 이름 유지(호환)
  const promptAndAddFolder = useCallback(async () => {
    try {
      // 1) 백엔드 폴더 선택 다이얼로그 호출
      const picked = await pickFolderPath();

      // 백엔드가 {path:"..."} 로 주든, 문자열로 주든 둘 다 처리
      const path = typeof picked === 'string' ? picked : (picked?.path ?? '');
      if (!path) return; // 사용자가 취소했으면 그냥 종료

      // 2) 표시 이름(선택)
      const defaultName = guessNameFromPath(path);
      const name = window.prompt('폴더 이름(표시용)을 입력하세요', defaultName) || defaultName;

      await createWatchedFolder({ name, path });
      await refresh();
    } catch (e) {
      // 폴더피커 미구현/오프라인이면 fallback
      const path = window.prompt('감시 폴더 경로를 입력하세요');
      if (!path) return;

      const defaultName = guessNameFromPath(path);
      const name = window.prompt('폴더 이름(표시용)을 입력하세요', defaultName) || defaultName;

      await createWatchedFolder({ name, path });
      await refresh();
    }
  }, [refresh]);

  const removeFolder = useCallback(async (id) => {
    await deleteWatchedFolder(id);
    await refresh();
  }, [refresh]);

  return { folders, loading, error, refresh, promptAndAddFolder, removeFolder };
}
