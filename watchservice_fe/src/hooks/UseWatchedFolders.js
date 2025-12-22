/**
 * 파일 이름 : UseWatchedFolders.js
 * 기능 : 감시 폴더 관리를 위한 커스텀 훅. 폴더 목록 조회, 추가, 삭제 기능을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import { useCallback, useEffect, useState } from 'react';
import {
  fetchWatchedFolders,
  pickFolderPath,
  createWatchedFolder,
  deleteWatchedFolder,
} from '../api/SettingApi';

/**
 * 함수 이름 : guessNameFromPath
 * 기능 : 경로 문자열에서 폴더 이름을 추출한다.
 * 매개변수 : path - 파일 경로 문자열
 * 반환값 : string - 추출된 폴더 이름
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function guessNameFromPath(path) {
  if (!path) return '폴더';
  const parts = String(path).split(/[/\\]+/).filter(Boolean);
  return parts.length ? parts[parts.length - 1] : '폴더';
}

/**
 * 함수 이름 : useWatchedFolders
 * 기능 : 감시 폴더 관리를 위한 커스텀 훅.
 * 매개변수 : 없음
 * 반환값 : Object - 폴더 목록, 로딩 상태, 에러, 새로고침 함수, 폴더 추가/삭제 함수를 포함한 객체
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
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
      // 폴더 선택 다이얼로그 실패 시에는 경로를 직접 입력받지 않고 에러만 알림
      console.error('pickFolderPath error:', e);
      alert('폴더 선택 다이얼로그 실행에 실패했습니다.\n백엔드 로그를 확인해 주세요.\n\n' + e.message);
    }
  }, [refresh]);

  const removeFolder = useCallback(async (id) => {
    await deleteWatchedFolder(id);
    await refresh();
  }, [refresh]);

  return { folders, loading, error, refresh, promptAndAddFolder, removeFolder };
}
