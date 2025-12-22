/**
 * 파일 이름 : App.js
 * 기능 : 애플리케이션의 루트 컴포넌트. 라우팅 설정과 전체 레이아웃을 관리한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import MainLayout from './layout/MainLayout';
import MainBoardPage from './pages/mainboard/MainBoardPage';

import NotificationPage from './pages/notifications/NotificationPage';
import NotificationDetailPage from './pages/notifications/NotificationDetailPage';
import NotificationStatsPage from './pages/notifications/NotificationStatsPage';

import LogsPage from './pages/logs/LogsPage';

import SettingHomePage from './pages/settings/SettingHomePage';
import SettingFoldersPage from './pages/settings/SettingFoldersPage';
import SettingNotifyPage from './pages/settings/SettingNotifyPage';
import SettingExceptionsPage from './pages/settings/SettingExceptionsPage';
import SettingResetPage from './pages/settings/SettingResetPage';
import SettingUpdatePage from './pages/settings/SettingUpdatePage';
import SettingFeedbackPage from './pages/settings/SettingFeedbackPage';

/**
 * 함수 이름 : App
 * 기능 : 애플리케이션의 루트 컴포넌트. 모든 라우트를 정의하고 MainLayout으로 감싼다.
 * 매개변수 : 없음
 * 반환값 : JSX.Element - 애플리케이션 루트 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function App() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          {/* 메인 보드 */}
          <Route path="/" element={<MainBoardPage />} />

          {/* 알림 */}
          <Route path="/notifications" element={<NotificationPage />} />
          <Route path="/notifications/stats" element={<NotificationStatsPage />} />
          <Route path="/notifications/:id" element={<NotificationDetailPage />} />

          {/* 로그 */}
          <Route path="/logs" element={<LogsPage />} />

          {/* 설정 */}
          <Route path="/settings" element={<SettingHomePage />} />
          <Route path="/settings/folders" element={<SettingFoldersPage />} />
          <Route path="/settings/exceptions" element={<SettingExceptionsPage />} />
          <Route path="/settings/notify" element={<SettingNotifyPage />} />
          <Route path="/settings/reset" element={<SettingResetPage />} />
          <Route path="/settings/update" element={<SettingUpdatePage />} />
          <Route path="/settings/feedback" element={<SettingFeedbackPage />} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  );
}

export default App;
