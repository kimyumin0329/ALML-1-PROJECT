/**
 * 파일 이름 : UseExceptions.js
 * 기능 : 예외 규칙 관리를 위한 커스텀 훅. 예외 규칙 목록 조회, 추가, 삭제 기능을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import { useEffect, useState } from 'react';
import {
  fetchExceptionRules,
  createExceptionRule,
  deleteExceptionRule,
} from '../api/SettingApi';

/**
 * 함수 이름 : useExceptions
 * 기능 : 예외 규칙 관리를 위한 커스텀 훅.
 * 매개변수 : 없음
 * 반환값 : Object - 예외 규칙 목록, 로딩 상태, 에러, 새로고침 함수, 예외 추가/삭제 함수를 포함한 객체
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function useExceptions() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const refresh = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await fetchExceptionRules();
      setItems(Array.isArray(data) ? data : (data?.items ?? []));
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const add = async (payload) => {
    await createExceptionRule(payload);
    await refresh();
  };

  const remove = async (id) => {
    await deleteExceptionRule(id);
    await refresh();
  };

  // 내부에서는 items/add/remove를 쓰되,
  // SettingExceptionsPage에서 기대하는 이름(exceptions/addException/removeException)도 함께 제공
  return {
    // 기본 필드
    items,
    loading,
    error,
    refresh,
    add,
    remove,
    // 별칭(페이지 용어에 맞춘 이름)
    exceptions: items,
    addException: add,
    removeException: remove,
  };
}
