/**
 * 파일 이름 : SettingHomePage.jsx
 * 기능 : 설정 메인 페이지. 각 설정 메뉴로 이동할 수 있는 카드 목록을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';
import { useNavigate } from 'react-router-dom';

/**
 * 함수 이름 : SettingHomePage
 * 기능 : 설정 메인 페이지 컴포넌트.
 * 매개변수 : 없음
 * 반환값 : JSX.Element - 설정 메인 페이지 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function SettingHomePage() {
  const navigate = useNavigate();

  const items = [
    { title: '감시 폴더', desc: '감시 대상 폴더 추가/삭제', path: '/settings/folders' },
    { title: '예외 관리', desc: '화이트리스트(감시 제외) 규칙', path: '/settings/exceptions' },
    // 예: SettingHomePage.jsx 내부 items 중
    { title: '알림 방식', desc: '팝업/소리 설정(백엔드 미구현 가능)', path: '/settings/notify' },
    { title: '초기화', desc: '설정/캐시 초기화(백엔드 미구현 가능)', path: '/settings/reset' },
    { title: '업데이트', desc: '버전/업데이트 확인(백엔드 미구현 가능)', path: '/settings/update' },
    { title: '문의/피드백', desc: '버그·문의 제출(백엔드 미구현 가능)', path: '/settings/feedback' },
  ];

  return (
    <div className="page-container">
      <h1>설정</h1>
      <p style={{ fontSize: 14, color: '#9ca3af', marginBottom: 16 }}>
        WatchService Agent 설정 메뉴입니다.
      </p>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: 12 }}>
        {items.map((it) => (
          <div
            key={it.path}
            className="card"
            onClick={() => navigate(it.path)}
            style={{
              cursor: 'pointer',
              padding: 14,
              border: '1px solid #e5e7eb',
              borderRadius: 10,
              background: 'white',
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 6 }}>{it.title}</div>
            <div style={{ fontSize: 13, color: '#6b7280' }}>{it.desc}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default SettingHomePage;
