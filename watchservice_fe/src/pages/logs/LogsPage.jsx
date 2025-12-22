/**
 * 파일 이름 : LogsPage.jsx
 * 기능 : 로그 관리 페이지. 로그 조회, 필터링, 삭제, 내보내기 기능을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React, { useMemo, useState } from 'react';
import { useLogs } from '../../hooks/UseLogs';
import LogFilterBar from '../../components/logs/LogFilterBar';
import LogTable from '../../components/logs/LogTable';
import LogDetailModal from '../../components/logs/LogDetailModal';
import { deleteLog, deleteLogs, exportLogs, fetchLogDetail } from '../../api/LogsApi';

/**
 * 함수 이름 : downloadText
 * 기능 : 텍스트 파일을 다운로드한다.
 * 매개변수 : filename - 파일명, text - 다운로드할 텍스트, mime - MIME 타입
 * 반환값 : 없음
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function downloadText(filename, text, mime = 'text/plain;charset=utf-8') {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * 함수 이름 : downloadJson
 * 기능 : JSON 파일을 다운로드한다.
 * 매개변수 : filename - 파일명, obj - 다운로드할 객체
 * 반환값 : 없음
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function downloadJson(filename, obj) {
  const text = JSON.stringify(obj, null, 2);
  downloadText(filename, text, 'application/json;charset=utf-8');
}

/**
 * 함수 이름 : LogsPage
 * 기능 : 로그 관리 페이지 컴포넌트. 로그 조회, 필터링, 삭제, 내보내기 기능을 제공한다.
 * 매개변수 : 없음
 * 반환값 : JSX.Element - 로그 관리 페이지 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function LogsPage() {
  const {
    logs,
    loading,
    error,

    page,
    setPage,
    limit,
    setLimit,
    total,

    filters,
    setFilters,

    refresh,
    search,
  } = useLogs(50);

  // ===== UI 입력(로컬 상태) : “검색” 눌렀을 때만 서버필터로 반영 =====
  const [keyword, setKeyword] = useState(filters.keyword || '');
  const [riskFilter, setRiskFilter] = useState(filters.aiLabel || 'ALL'); // ALL | SAFE | WARNING | DANGER | UNKNOWN
  const [from, setFrom] = useState(filters.from || '');
  const [to, setTo] = useState(filters.to || '');
  const [sort, setSort] = useState(filters.sort || 'collectedAt,desc');

  const [selectedLog, setSelectedLog] = useState(null);
  const [selectedIds, setSelectedIds] = useState(() => new Set());

  const totalPages = useMemo(() => {
    if (typeof total !== 'number' || total < 0) return null;
    return Math.max(1, Math.ceil(total / (limit || 1)));
  }, [total, limit]);

  const handleLimitChange = (e) => {
    const value = Number(e.target.value) || 10;
    setLimit(value);
  };

  // UNKNOWN(라벨 없음)은 백엔드 필터가 정의되어 있지 않아서, 그 경우만 로컬로 거른다.
  const visibleLogs = useMemo(() => {
    const list = logs || [];

    if (riskFilter === 'UNKNOWN') {
      return list.filter((l) => !l.aiLabel);
    }
    return list;
  }, [logs, riskFilter]);

  const toggleSelect = (id) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleSelectAll = () => {
    setSelectedIds((prev) => {
      const all = visibleLogs.map((l) => l.id);
      const allSelected = all.length > 0 && all.every((id) => prev.has(id));
      if (allSelected) return new Set();
      return new Set(all);
    });
  };

  const applySearch = () => {
    const aiLabel =
      riskFilter === 'SAFE' || riskFilter === 'WARNING' || riskFilter === 'DANGER'
        ? riskFilter
        : ''; // ALL/UNKNOWN은 서버로 aiLabel을 안 보냄

    setFilters({
      ...filters,
      keyword: keyword || '',
      from: from || '',
      to: to || '',
      sort: sort || 'collectedAt,desc',
      aiLabel,
    });

    // page=0으로 리셋 (로드는 useEffect가 처리)
    search();
  };

  // ✅ 명세 14번: 필터 초기화
  const resetFilters = () => {
    // 1) UI 입력값 초기화
    setKeyword('');
    setRiskFilter('ALL');
    setFrom('');
    setTo('');
    setSort('collectedAt,desc');

    // 2) 서버 필터 초기화
    setFilters({
      ...filters,
      keyword: '',
      from: '',
      to: '',
      sort: 'collectedAt,desc',
      aiLabel: '',
      eventType: '',
    });

    // 3) 페이지 리셋 → useEffect가 재조회
    search();
  };

  const handleRowClick = async (log) => {
    try {
      const detail = await fetchLogDetail(log.id);
      setSelectedLog(detail || log);
    } catch (e) {
      // 상세 조회 실패 시에도 리스트 데이터로 모달은 열어주자
      setSelectedLog(log);
    }
  };

  const handleDeleteOne = async (id) => {
    if (!window.confirm(`로그(ID=${id})를 삭제할까요?`)) return;

    try {
      await deleteLog(id);
      setSelectedIds((prev) => {
        const next = new Set(prev);
        next.delete(id);
        return next;
      });
      setSelectedLog(null);
      refresh();
    } catch (e) {
      alert(`삭제 실패: ${e.message}`);
    }
  };

  const handleDeleteSelected = async () => {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return alert('선택된 로그가 없습니다.');
    if (!window.confirm(`선택한 로그 ${ids.length}개를 삭제할까요?`)) return;

    try {
      await deleteLogs(ids);
      setSelectedIds(new Set());
      setSelectedLog(null);
      refresh();
    } catch (e) {
      alert(`삭제 실패: ${e.message}`);
    }
  };

  const handleExport = async (format) => {
    const ids = Array.from(selectedIds);

    const req = {
      format: format === 'JSON' ? 'JSON' : 'CSV',
      ids: ids.length > 0 ? ids : [],
      // ids가 비어있으면 filters 기반 전체 내보내기 가능하도록 filters 전달
      filters:
        ids.length === 0
          ? {
              page,
              size: limit,
              from: filters.from || '',
              to: filters.to || '',
              keyword: filters.keyword || '',
              aiLabel: filters.aiLabel || '',
              eventType: filters.eventType || '',
              sort: filters.sort || '',
            }
          : null,
    };

    try {
      const res = await exportLogs(req);

      if (format === 'JSON') {
        downloadJson(`logs_${Date.now()}.json`, res);
        return;
      }

      downloadText(`logs_${Date.now()}.csv`, String(res || ''), 'text/csv;charset=utf-8');
    } catch (e) {
      alert(`내보내기 실패: ${e.message}`);
    }
  };

  return (
    <div className="page-container">
      <div>
        <h1>로그 관리</h1>
        <p style={{ fontSize: 13, color: '#9ca3af', marginTop: 4, marginBottom: 8 }}>
          감시 기능이 수집한 원시 파일 이벤트와, 그로 인해 파일 크기/엔트로피/확장자가 어떻게 바뀌었는지(AI 분석 전 단계)를 시간순으로 보여줍니다.
        </p>
      </div>

      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
        <label>
          표시 개수:&nbsp;
          <select value={limit} onChange={handleLimitChange}>
            <option value={10}>10</option>
            <option value={30}>30</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
          </select>
        </label>

        <button className="btn" onClick={() => handleExport('CSV')}>
          CSV 내보내기
        </button>
        <button className="btn" onClick={() => handleExport('JSON')}>
          JSON 내보내기
        </button>

        <button className="btn" onClick={handleDeleteSelected} disabled={selectedIds.size === 0}>
          선택 삭제
        </button>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <button className="btn" disabled={page <= 0} onClick={() => setPage(page - 1)}>
            이전
          </button>
          <span>
            페이지: {page + 1}
            {totalPages ? ` / ${totalPages}` : ''}
          </span>
          <button
            className="btn"
            disabled={totalPages ? page + 1 >= totalPages : false}
            onClick={() => setPage(page + 1)}
          >
            다음
          </button>
        </div>
      </div>

      <LogFilterBar
        keyword={keyword}
        setKeyword={setKeyword}
        riskFilter={riskFilter}
        setRiskFilter={setRiskFilter}
        from={from}
        setFrom={setFrom}
        to={to}
        setTo={setTo}
        sort={sort}
        setSort={setSort}
        onSearch={applySearch}
        onReset={resetFilters}
        onRefresh={refresh}
      />

      {loading && <p>로그를 불러오는 중...</p>}
      {error && <p style={{ color: 'red' }}>오류: {error.message}</p>}

      {!loading && !error && (
        <LogTable
          logs={visibleLogs}
          onRowClick={handleRowClick}
          selectedIds={selectedIds}
          onToggleSelect={toggleSelect}
          onToggleSelectAll={toggleSelectAll}
        />
      )}

      <LogDetailModal
        log={selectedLog}
        onClose={() => setSelectedLog(null)}
        onDelete={() => handleDeleteOne(selectedLog?.id)}
      />
    </div>
  );
}

export default LogsPage;
