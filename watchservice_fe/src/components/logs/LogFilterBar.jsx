// src/components/logs/LogFilterBar.jsx
import React from 'react';

/**
 * LogFilterBar (명세형)
 * - keyword: 키워드(파일경로/이벤트/AI상세)
 * - riskFilter: ALL | DANGER | WARNING | SAFE | UNKNOWN
 * - from/to: YYYY-MM-DD (기간)
 * - sort: "collectedAt,desc" 같은 문자열
 *
 * onSearch: 검색 적용(서버 /logs 구현되면 서버필터로 동작)
 * onReset: 필터 초기화(명세 14번)
 * onRefresh: 새로고침
 */
function LogFilterBar({
  keyword,
  setKeyword,

  riskFilter,
  setRiskFilter,

  from,
  setFrom,
  to,
  setTo,

  sort,
  setSort,

  onSearch,
  onReset,
  onRefresh,

  // 하위호환: 예전 props 이름으로 넘겨도 동작하게
  refreshLogs,
}) {
  const handleKeywordChange = (e) => setKeyword?.(e.target.value);
  const handleRiskChange = (e) => setRiskFilter?.(e.target.value);

  const handleFromChange = (e) => setFrom?.(e.target.value);
  const handleToChange = (e) => setTo?.(e.target.value);

  const handleSortChange = (e) => setSort?.(e.target.value);

  const handleSearchClick = () => onSearch?.();
  const handleResetClick = () => onReset?.();
  const handleRefreshClick = () => (onRefresh ? onRefresh() : refreshLogs?.());

  return (
    <div className="log-filter-bar" style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      <input
        type="text"
        placeholder="파일 경로 / 이벤트 / AI 상세 검색"
        value={keyword || ''}
        onChange={handleKeywordChange}
      />

      <select value={riskFilter || 'ALL'} onChange={handleRiskChange}>
        <option value="ALL">위험도 전체</option>
        <option value="DANGER">DANGER</option>
        <option value="WARNING">WARNING</option>
        <option value="SAFE">SAFE</option>
        <option value="UNKNOWN">UNKNOWN/기타</option>
      </select>

      <input type="date" value={from || ''} onChange={handleFromChange} />
      <input type="date" value={to || ''} onChange={handleToChange} />

      <select value={sort || 'collectedAt,desc'} onChange={handleSortChange}>
        <option value="collectedAt,desc">시간(최신순)</option>
        <option value="collectedAt,asc">시간(오래된순)</option>
        <option value="aiScore,desc">AI 점수(높은순)</option>
        <option value="aiScore,asc">AI 점수(낮은순)</option>
        <option value="entropy,desc">엔트로피(높은순)</option>
        <option value="entropy,asc">엔트로피(낮은순)</option>
      </select>

      <button className="btn" onClick={handleSearchClick}>
        검색
      </button>

      <button className="btn" onClick={handleResetClick}>
        필터 초기화
      </button>

      <button className="btn" onClick={handleRefreshClick}>
        새로고침
      </button>
    </div>
  );
}

export default LogFilterBar;
