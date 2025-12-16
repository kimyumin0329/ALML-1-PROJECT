// src/pages/settings/SettingExceptionsPage.jsx
import React, { useState } from 'react';
import { useExceptions } from '../../hooks/UseExceptions';

function SettingExceptionsPage() {
  const { exceptions, loading, error, refresh, addException, removeException } = useExceptions();

  const [type, setType] = useState('PATH');
  const [pattern, setPattern] = useState('');
  const [memo, setMemo] = useState('');

  const handleAdd = async (e) => {
    e.preventDefault();

    if (!pattern.trim()) {
      alert('예외로 등록할 경로나 패턴을 입력해주세요.');
      return;
    }

    await addException({ type, pattern: pattern.trim(), memo: memo.trim() });
    setPattern('');
    setMemo('');
  };

  const handleRemove = (id) => {
    if (!window.confirm('이 예외 규칙을 삭제하시겠습니까?')) return;
    removeException(id);
  };

  return (
    <div className="page-container">
      <h1>예외(화이트리스트) 설정</h1>
      <p style={{ fontSize: 14, color: '#9ca3af', marginBottom: 16 }}>
        감시 대상에서 제외할 파일/폴더/확장자를 등록합니다.
      </p>

      <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <button className="btn" onClick={refresh} disabled={loading}>
          새로고침
        </button>
        {loading && <span style={{ fontSize: 13, color: '#9ca3af' }}>불러오는 중...</span>}
        {error && (
          <span style={{ fontSize: 13, color: 'red' }}>
            서버 연동 오류: {error.message}
          </span>
        )}
      </div>

      {/* 예외 추가 폼 */}
      <form
        onSubmit={handleAdd}
        style={{
          display: 'flex',
          flexDirection: 'column',
          gap: 8,
          maxWidth: 480,
          marginBottom: 24,
        }}
      >
        <label style={{ fontSize: 13 }}>
          예외 종류
          <select
            value={type}
            onChange={(e) => setType(e.target.value)}
            style={{ marginLeft: 8 }}
          >
            <option value="PATH">경로(파일/폴더)</option>
            <option value="EXT">확장자</option>
          </select>
        </label>

        <label style={{ fontSize: 13 }}>
          패턴
          <input
            type="text"
            value={pattern}
            onChange={(e) => setPattern(e.target.value)}
            placeholder={type === 'PATH' ? '/Users/~/Downloads/temp' : '.log, .tmp 처럼 확장자'}
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>

        <label style={{ fontSize: 13 }}>
          메모(선택)
          <input
            type="text"
            value={memo}
            onChange={(e) => setMemo(e.target.value)}
            placeholder="예: 백업 폴더, 로그 폴더 등"
            style={{ width: '100%', marginTop: 4 }}
          />
        </label>

        <button type="submit" className="btn" style={{ alignSelf: 'flex-start' }}>
          예외 추가
        </button>
      </form>

      {/* 예외 목록 */}
      <div className="exception-list-panel">
        <h2 style={{ fontSize: 16, marginBottom: 8 }}>등록된 예외 규칙</h2>

        {exceptions.length === 0 && (
          <p style={{ fontSize: 13, color: '#9ca3af' }}>
            아직 등록된 예외 규칙이 없습니다.
          </p>
        )}

        {exceptions.map((ex) => (
          <div
            key={ex.id}
            className="exception-item"
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '8px 12px',
              borderRadius: 6,
              border: '1px solid #e5e7eb',
              marginBottom: 6,
            }}
          >
            <div>
              <div style={{ fontSize: 13, fontWeight: 500 }}>
                [{ex.type}] {ex.pattern}
              </div>
              {ex.memo && (
                <div style={{ fontSize: 12, color: '#9ca3af', marginTop: 2 }}>
                  {ex.memo}
                </div>
              )}
            </div>
            <button className="btn-icon" onClick={() => handleRemove(ex.id)}>
              삭제
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}

export default SettingExceptionsPage;
