// src/pages/settings/SettingUpdatePage.jsx

import React from 'react';

function SettingUpdatePage() {
  const currentVersion = '0.1.0-dev';

  const handleCheckUpdate = () => {
    alert('업데이트 서버와 통신해서 최신 버전을 확인하는 기능을 나중에 구현할 수 있습니다.');
  };

  return (
    <div className="page-container">
      <h1>버전 / 업데이트 정보</h1>
      <p className="page-description">
        현재 WatchService Agent의 버전과 향후 업데이트 기능을 확인하는 화면입니다.
      </p>

      <div className="card">
        <div className="card-row">
          <span className="card-label">현재 버전</span>
          <span className="card-value">{currentVersion}</span>
        </div>
        <div className="card-row">
          <button className="btn" onClick={handleCheckUpdate}>
            업데이트 확인
          </button>
        </div>
      </div>
    </div>
  );
}

export default SettingUpdatePage;
