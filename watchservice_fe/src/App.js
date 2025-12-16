import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import MainLayout from './layout/MainLayout';
import MainBoardPage from './pages/mainboard/MainBoardPage';

// 알림 관련 페이지
import NotificationPage from './pages/notifications/NotificationPage';
import NotificationDetailPage from './pages/notifications/NotificationDetailPage';
import NotificationStatsPage from './pages/notifications/NotificationStatsPage';

// 로그 페이지
import LogsPage from './pages/logs/LogsPage';

// 설정 관련 페이지
import SettingHomePage from './pages/settings/SettingHomePage';
import SettingFoldersPage from './pages/settings/SettingFoldersPage';
import SettingNotifyPage from './pages/settings/SettingNotifyPage';

import SettingExceptionsPage from './pages/settings/SettingExceptionsPage';
import SettingResetPage from './pages/settings/SettingResetPage';
import SettingUpdatePage from './pages/settings/SettingUpdatePage';
import SettingFeedbackPage from './pages/settings/SettingFeedbackPage';

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
