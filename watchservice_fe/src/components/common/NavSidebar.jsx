/**
 * 파일 이름 : NavSidebar.jsx
 * 기능 : 네비게이션 사이드바 컴포넌트. 메인 보드, 알림, 로그, 설정 메뉴를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';
import { NavLink } from 'react-router-dom';

/**
 * 함수 이름 : NavSidebar
 * 기능 : 네비게이션 사이드바 컴포넌트.
 * 매개변수 : 없음
 * 반환값 : JSX.Element - 네비게이션 사이드바 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function NavSidebar() {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <span className="sidebar-logo-main">WatchService</span>
        <span className="sidebar-logo-sub">Desktop Agent</span>
      </div>

      <nav className="sidebar-nav">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            'sidebar-link' + (isActive ? ' sidebar-link-active' : '')
          }
        >
          메인 보드
        </NavLink>

        <NavLink
          to="/notifications"
          className={({ isActive }) =>
            'sidebar-link' + (isActive ? ' sidebar-link-active' : '')
          }
        >
          알림
        </NavLink>

        <NavLink
          to="/logs"
          className={({ isActive }) =>
            'sidebar-link' + (isActive ? ' sidebar-link-active' : '')
          }
        >
          로그
        </NavLink>

        <NavLink
          to="/settings"
          className={({ isActive }) =>
            'sidebar-link' + (isActive ? ' sidebar-link-active' : '')
          }
        >
          설정
        </NavLink>
      </nav>
    </aside>
  );
}

export default NavSidebar;
