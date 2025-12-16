import React from 'react';
import NavSidebar from '../components/common/NavSidebar';

function MainLayout({ children }) {
  return (
    <div className="app-root">
      {/* 상단 제목바 느낌 (나중에 윈도우 컨트롤 버튼 넣을 자리) */}
      <header className="app-header">
        <div className="app-title">WatchService Agent</div>
        <div className="app-header-buttons">
          {/* 여긴 나중에 최소화/닫기 버튼 같은 것 표현 가능 */}
        </div>
      </header>

      {/* 본문: 좌측 메뉴 + 우측 콘텐츠 */}
      <div className="app-body">
        <NavSidebar />
        <main className="app-content">
          {children}
        </main>
      </div>
    </div>
  );
}

export default MainLayout;
