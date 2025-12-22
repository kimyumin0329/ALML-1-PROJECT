/**
 * 파일 이름 : SettingResetPage.jsx
 * 기능 : 설정 초기화 페이지. 로컬 설정과 서버 설정을 기본값으로 되돌린다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React, { useState } from 'react';
import { resetSettings } from '../../api/SettingApi';

/**
 * 함수 이름 : SettingResetPage
 * 기능 : 설정 초기화 페이지 컴포넌트.
 * 매개변수 : 없음
 * 반환값 : JSX.Element - 설정 초기화 페이지 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function SettingResetPage() {
  const [loading, setLoading] = useState(false);
  const [note, setNote] = useState('');

  const handleReset = async () => {
    const ok = window.confirm('설정을 기본값으로 초기화할까요?');
    if (!ok) return;

    setLoading(true);
    setNote('');

    // 로컬 캐시/설정 초기화
    try {
      localStorage.removeItem('watchservice.watchedFolders');
      localStorage.removeItem('watchservice.exceptions');
      localStorage.removeItem('watchservice.notifySettings');
    } catch {
      // ignore
    }

    // 서버 초기화(있으면) 시도
    try {
      await resetSettings();
      setNote('초기화 완료(서버 반영).');
    } catch (e) {
      setNote('초기화 완료(로컬 초기화). 서버는 아직 미구현/오프라인일 수 있습니다.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <h1>초기화</h1>
      <p style={{ fontSize: 14, color: '#9ca3af', marginBottom: 16 }}>
        설정을 기본값으로 되돌립니다. (서버가 없으면 로컬 기준으로 초기화됩니다)
      </p>

      <div style={{ padding: 14, border: '1px solid #fee2e2', background: '#fff1f2', borderRadius: 10, maxWidth: 560 }}>
        <strong style={{ display: 'block', marginBottom: 8 }}>주의</strong>
        <div style={{ fontSize: 13 }}>
          이 작업은 되돌릴 수 없습니다. 필요한 경우 내보내기/백업 후 진행하세요.
        </div>

        <button className="btn btn-danger" style={{ marginTop: 12 }} onClick={handleReset} disabled={loading}>
          {loading ? '초기화 중...' : '초기화'}
        </button>

        {note && <p style={{ marginTop: 10, fontSize: 13 }}>{note}</p>}
      </div>
    </div>
  );
}

export default SettingResetPage;
