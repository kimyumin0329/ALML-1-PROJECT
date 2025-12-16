// src/components/logs/LogTable.jsx
import React, { useMemo } from 'react';

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
            <th>크기</th>
            <th>AI</th>
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
              <td title={log.path} style={{ maxWidth: 520, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                {log.path}
              </td>
              <td>{log.size}</td>
              <td>{log.aiLabel || '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default LogTable;
