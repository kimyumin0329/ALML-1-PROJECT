/**
 * 파일 이름 : LogFilterBar.jsx
 * 기능 : 로그 필터 바 컴포넌트. 키워드 검색, 위험도 필터, 날짜 범위, 정렬 옵션을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';

/**
 * 함수 이름 : LogFilterBar
 * 기능 : 로그 필터 바 컴포넌트.
 * 매개변수 : keyword - 검색 키워드, setKeyword - 키워드 설정 함수, riskFilter - 위험도 필터 (ALL|DANGER|WARNING|SAFE|UNKNOWN), setRiskFilter - 위험도 필터 설정 함수, from - 시작 날짜 (YYYY-MM-DD), setFrom - 시작 날짜 설정 함수, to - 종료 날짜 (YYYY-MM-DD), setTo - 종료 날짜 설정 함수, sort - 정렬 기준, setSort - 정렬 기준 설정 함수, onSearch - 검색 핸들러, onReset - 필터 초기화 핸들러, onRefresh - 새로고침 핸들러, refreshLogs - 새로고침 함수 (하위 호환)
 * 반환값 : JSX.Element - 로그 필터 바 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
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
        placeholder="파일 경로 / 이벤트"
        value={keyword || ''}
        onChange={handleKeywordChange}
      />

      {/* 
      <select value={riskFilter || 'ALL'} onChange={handleRiskChange}>
        <option value="ALL">위험도 전체</option>
        <option value="DANGER">DANGER</option>
        <option value="WARNING">WARNING</option>
        <option value="SAFE">SAFE</option>
        <option value="UNKNOWN">UNKNOWN/기타</option>
      </select> 
      */}

      <input type="date" value={from || ''} onChange={handleFromChange} />
      <input type="date" value={to || ''} onChange={handleToChange} />

      <select value={sort || 'collectedAt,desc'} onChange={handleSortChange}>
        <option value="collectedAt,desc">시간(최신순)</option>
        <option value="collectedAt,asc">시간(오래된순)</option>
        {/* <option value="aiScore,desc">AI 점수(높은순)</option>
        <option value="aiScore,asc">AI 점수(낮은순)</option> */}
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
