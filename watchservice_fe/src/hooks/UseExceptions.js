import { useEffect, useState } from 'react';
import {
  fetchExceptionRules,
  createExceptionRule,
  deleteExceptionRule,
} from '../api/SettingApi';

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
