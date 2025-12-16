// src/hooks/UseProtectionStatus.js
import { useCallback, useEffect, useState } from 'react';
import { fetchDashboardSummary } from '../api/DashboardApi';

export function useProtectionStatus() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchDashboardSummary();
      setData(res);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  return {
    summary: data,
    loading,
    error,
    refresh: load,
  };
}
