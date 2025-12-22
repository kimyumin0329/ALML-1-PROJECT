/**
 * 파일 이름 : MainLayout.jsx
 * 기능 : 메인 레이아웃 컴포넌트. 헤더, 사이드바, 메인 콘텐츠 영역을 구성한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';
import NavSidebar from '../components/common/NavSidebar';

/**
 * 함수 이름 : MainLayout
 * 기능 : 메인 레이아웃 컴포넌트.
 * 매개변수 : children - 자식 컴포넌트
 * 반환값 : JSX.Element - 메인 레이아웃 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
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
