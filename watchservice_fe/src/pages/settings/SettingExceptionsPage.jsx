/**
 * 파일 이름 : SettingExceptionsPage.jsx
 * 기능 : 예외(화이트리스트) 규칙을 관리하는 페이지. 경로나 확장자 기반 예외를 등록/삭제할 수 있다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React, { useState } from 'react';
import { useExceptions } from '../../hooks/UseExceptions';
import { pickFolderPath } from '../../api/SettingApi';

/**
 * 함수 이름 : SettingExceptionsPage
 * 기능 : 예외 규칙 관리 페이지 컴포넌트. PATH 타입일 때는 폴더 선택 다이얼로그를 사용할 수 있다.
 * 매개변수 : 없음
 * 반환값 : JSX.Element - 예외 규칙 관리 페이지 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function SettingExceptionsPage() {
  const { exceptions, loading, error, refresh, addException, removeException } = useExceptions();

  const [type, setType] = useState('PATH');
  const [pattern, setPattern] = useState('');
  const [memo, setMemo] = useState('');
  const [pickingFolder, setPickingFolder] = useState(false);

  /**
   * 함수 이름 : handlePickFolder
   * 기능 : 백엔드 폴더 선택 다이얼로그를 호출하여 폴더 경로를 선택한다.
   * 매개변수 : 없음
   * 반환값 : 없음
   * 작성 날짜 : 2025/12/17
   * 작성자 : 시스템
   */
  const handlePickFolder = async () => {
    try {
      setPickingFolder(true);
      const picked = await pickFolderPath();
      
      // 백엔드가 {path:"..."} 로 주든, 문자열로 주든 둘 다 처리
      const path = typeof picked === 'string' ? picked : (picked?.path ?? '');
      if (path) {
        setPattern(path);
      }
    } catch (e) {
      // 폴더 피커 미구현/오프라인이면 fallback
      const path = window.prompt('예외로 등록할 폴더 경로를 입력하세요');
      if (path) {
        setPattern(path);
      }
    } finally {
      setPickingFolder(false);
    }
  };

  /**
   * 함수 이름 : handleAdd
   * 기능 : 예외 규칙을 추가한다.
   * 매개변수 : e - 폼 제출 이벤트
   * 반환값 : 없음
   * 작성 날짜 : 2025/12/17
   * 작성자 : 시스템
   */
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

  /**
   * 함수 이름 : handleRemove
   * 기능 : 예외 규칙을 삭제한다.
   * 매개변수 : id - 삭제할 예외 규칙 ID
   * 반환값 : 없음
   * 작성 날짜 : 2025/12/17
   * 작성자 : 시스템
   */
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
          <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
            <input
              type="text"
              value={pattern}
              onChange={type === 'EXT' ? (e) => setPattern(e.target.value) : undefined}
              placeholder={
                type === 'PATH'
                  ? '폴더 선택 버튼을 눌러 경로를 선택하세요'
                  : '.log, .tmp 처럼 확장자'
              }
              readOnly={type === 'PATH'}
              style={{
                flex: 1,
                backgroundColor: type === 'PATH' ? '#020617' : undefined,
                cursor: type === 'PATH' ? 'default' : 'text',
              }}
            />
            {type === 'PATH' && (
              <button
                type="button"
                className="btn"
                onClick={handlePickFolder}
                disabled={pickingFolder}
                style={{ whiteSpace: 'nowrap' }}
              >
                {pickingFolder ? '선택 중...' : '폴더 선택'}
              </button>
            )}
          </div>
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
