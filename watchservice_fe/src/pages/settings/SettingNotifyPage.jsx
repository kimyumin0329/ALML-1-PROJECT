/**
 * 파일 이름 : SettingNotifyPage.jsx
 * 기능 : 알림 방식 설정 페이지. 팝업/소리 알림을 설정할 수 있다. 백엔드가 없으면 로컬 저장소를 사용한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React, { useEffect, useState } from 'react';
import { fetchNotificationSettings, updateNotificationSettings } from '../../api/SettingApi';

const STORAGE_KEY = 'watchservice.notifySettings';

function SettingNotifyPage() {
  const [popup, setPopup] = useState(true);
  const [sound, setSound] = useState(false);

  const [loading, setLoading] = useState(false);
  const [note, setNote] = useState('');

  useEffect(() => {
    let mounted = true;

    (async () => {
      setLoading(true);
      setNote('');
      try {
        const s = await fetchNotificationSettings();
        if (!mounted) return;

        setPopup(!!s.popup);
        setSound(!!s.sound);

        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ popup: !!s.popup, sound: !!s.sound })
        );
        setNote('서버 설정을 불러왔습니다.');
      } catch (e) {
        try {
          const raw = localStorage.getItem(STORAGE_KEY);
          if (raw) {
            const v = JSON.parse(raw);
            setPopup(!!v.popup);
            setSound(!!v.sound);
            setNote('서버가 없어 로컬 설정을 사용 중입니다.');
          } else {
            setNote('서버가 없어 기본값(로컬)으로 동작합니다.');
          }
        } catch {
          setNote('서버가 없어 기본값(로컬)으로 동작합니다.');
        }
      } finally {
        if (mounted) setLoading(false);
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);

  const handleSave = async () => {
    setLoading(true);
    setNote('');
    const payload = { popup, sound };

    // 로컬 먼저 저장
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload));

    // 서버 있으면 저장 시도
    try {
      await updateNotificationSettings(payload);
      setNote('저장 완료(서버 반영).');
    } catch (e) {
      setNote('저장 완료(로컬 저장). 서버는 아직 미구현/오프라인일 수 있습니다.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="page-container">
      <h1>알림 방식 설정</h1>
      <p style={{ fontSize: 14, color: '#9ca3af', marginBottom: 16 }}>
        팝업/소리 알림 방식을 선택합니다. (백엔드가 없으면 로컬 저장으로 동작)
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', gap: 10, maxWidth: 420 }}>
        <label style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <input type="checkbox" checked={popup} onChange={(e) => setPopup(e.target.checked)} />
          팝업 알림
        </label>

        <label style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <input type="checkbox" checked={sound} onChange={(e) => setSound(e.target.checked)} />
          소리 알림
        </label>

        <button className="btn" onClick={handleSave} disabled={loading}>
          {loading ? '처리 중...' : '저장'}
        </button>

        {note && <p style={{ fontSize: 13, color: '#6b7280' }}>{note}</p>}
      </div>
    </div>
  );
}

export default SettingNotifyPage;
