/**
 * 파일 이름 : LogTable.jsx
 * 기능 : 로그 목록을 테이블 형태로 표시하는 컴포넌트. 체크박스 선택, 행 클릭 이벤트를 지원한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React, { useMemo } from 'react';

/**
 * 함수 이름 : LogTable
 * 기능 : 로그 목록을 테이블로 렌더링한다.
 * 매개변수 : logs - 로그 리스트, onRowClick - 행 클릭 핸들러, selectedIds - 선택된 로그 ID Set, onToggleSelect - 개별 선택 토글 핸들러, onToggleSelectAll - 전체 선택 토글 핸들러
 * 반환값 : JSX.Element - 로그 테이블 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function LogTable({ logs, onRowClick, selectedIds, onToggleSelect, onToggleSelectAll }) {
  const list = logs || [];
  const allSelected = useMemo(() => {
    if (!list.length) return false;
    return list.every((l) => selectedIds?.has(l.id));
  }, [list, selectedIds]);

  if (!list.length) return <p>표시할 로그가 없습니다.</p>;

  return (
    <div className="log-table-wrapper">
      <table className="log-table">
        <thead>
          <tr>
            <th style={{ width: 36 }}>
              <input
                type="checkbox"
                checked={allSelected}
                onChange={() => onToggleSelectAll?.()}
              />
            </th>
            <th>ID</th>
            <th>수집 시각</th>
            <th>이벤트</th>
            <th>경로</th>
            <th>크기 변화</th>
            <th>엔트로피 변화</th>
            {/* <th>AI</th> */}
          </tr>
        </thead>

        <tbody>
          {list.map((log) => (
            <tr key={log.id} onClick={() => onRowClick?.(log)} style={{ cursor: 'pointer' }}>
              <td onClick={(e) => e.stopPropagation()}>
                <input
                  type="checkbox"
                  checked={selectedIds?.has(log.id) || false}
                  onChange={() => onToggleSelect?.(log.id)}
                />
              </td>
              <td>{log.id}</td>
              <td>{log.collectedAt}</td>
              <td>{log.eventType}</td>
              <td title={log.path} style={{ maxWidth: 420, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {log.path}
              </td>
              <td>
                {log.sizeBefore != null ? `${log.sizeBefore} → ${log.sizeAfter ?? log.size}` : (log.size ?? '-')}
                {log.sizeDiff != null ? ` (${log.sizeDiff >= 0 ? '+' : ''}${log.sizeDiff})` : ''}
              </td>
              <td>
                {log.entropyBefore != null ? `${log.entropyBefore} → ${log.entropyAfter ?? log.entropy ?? '-'}` : (log.entropy ?? '-')}
                {log.entropyDiff != null ? ` (${log.entropyDiff >= 0 ? '+' : ''}${log.entropyDiff.toFixed(4)})` : ''}
              </td>
              {/* <td>{log.aiLabel || '-'}</td> */}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default LogTable;
