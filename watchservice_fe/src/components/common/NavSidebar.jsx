import React from 'react';
import { NavLink } from 'react-router-dom';

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
