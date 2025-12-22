/**
 * 파일 이름 : NotificationPage.jsx
 * 기능 : 알림 히스토리 페이지. 알림 목록 조회, 필터링, 상세 보기 기능을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import React, { useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useNotifications } from '../../hooks/UseNotifications';

/**
 * 함수 이름 : NotificationPage
 * 기능 : 알림 히스토리 페이지 컴포넌트. 알림 목록을 표시하고 필터링 및 상세 보기를 지원한다.
 * 매개변수 : 없음
 * 반환값 : JSX.Element - 알림 히스토리 페이지 컴포넌트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
function NotificationPage() {
  const navigate = useNavigate();

  const {
    notifications,
    total,
    loading,
    error,

    page,
    setPage,

    limit,
    setLimit,

    keyword,
    setKeyword,
    level,
    setLevel,
    from,
    setFrom,
    to,
    setTo,
    sort,
    setSort,

    search,
    refresh,
  } = useNotifications(50);

  const totalPages = useMemo(() => {
    const size = Number(limit) || 20;
    return Math.max(1, Math.ceil((Number(total) || 0) / size));
  }, [total, limit]);

  const handleClickItem = (item) => {
    navigate(`/notifications/${item.id}`, {
      state: { notification: item },
    });
  };

  const handleSearch = () => {
    search();
  };

  return (
    <div className="page-container">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <div>
          <h1>알림 히스토리</h1>
          <p style={{ fontSize: 13, color: '#9ca3af', marginTop: 4 }}>
            최근 3초 윈도우 동안의 행동 패턴을 AI가 분석한 결과를 요약 경보로 제공합니다. 각 알림은 윈도우 단위이며, 영향 받은 파일 경로 목록을 포함합니다.
          </p>
        </div>
        <button className="btn" onClick={() => navigate('/notifications/stats')}>
          통계 보기
        </button>
      </div>

      {/* 필터 바 */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center', marginBottom: 12 }}>
        <label>
          표시 개수:&nbsp;
          <select value={limit} onChange={(e) => setLimit(e.target.value)}>
            <option value={20}>20</option>
            <option value={50}>50</option>
            <option value={100}>100</option>
            <option value={200}>200</option>
          </select>
        </label>

        <select value={level} onChange={(e) => setLevel(e.target.value)}>
          <option value="ALL">위험도 전체</option>
          <option value="DANGER">DANGER</option>
          <option value="WARNING">WARNING</option>
          <option value="SAFE">SAFE</option>
        </select>

        <input
          type="date"
          value={from || ''}
          onChange={(e) => setFrom(e.target.value)}
        />
        <input
          type="date"
          value={to || ''}
          onChange={(e) => setTo(e.target.value)}
        />

        <input
          type="text"
          placeholder="경로/AI 상세/패밀리 검색"
          value={keyword || ''}
          onChange={(e) => setKeyword(e.target.value)}
          style={{ minWidth: 260 }}
        />

        <select value={sort} onChange={(e) => setSort(e.target.value)}>
          <option value="createdAt,desc">시간(최신순)</option>
          <option value="createdAt,asc">시간(오래된순)</option>
          <option value="aiScore,desc">AI 점수(높은순)</option>
          <option value="aiScore,asc">AI 점수(낮은순)</option>
        </select>

        <button className="btn" onClick={handleSearch}>
          검색
        </button>
        <button className="btn" onClick={refresh}>
          새로고침
        </button>

        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
          <button className="btn" disabled={page <= 0} onClick={() => setPage(page - 1)}>
            이전
          </button>
          <span>
            페이지: {page + 1} / {totalPages} (총 {total}건)
          </span>
          <button className="btn" disabled={page + 1 >= totalPages} onClick={() => setPage(page + 1)}>
            다음
          </button>
        </div>
      </div>

      {loading && <p>불러오는 중...</p>}
      {error && <p style={{ color: 'red' }}>알림 로드 오류: {error.message}</p>}

      {!loading && !error && notifications.length === 0 && <p>표시할 알림이 없습니다.</p>}

      {!loading && !error && notifications.length > 0 && (
        <ul className="notification-list">
          {notifications.map((item) => (
            <li
              key={item.id}
              className="notification-item"
              onClick={() => handleClickItem(item)}
              style={{ cursor: 'pointer' }}
            >
              <div className="notification-item-main">
                <span className="notification-title">
                  [{item.aiLabel || 'UNKNOWN'}] {item.topFamily || '분류 중'}
                </span>
                <span className="notification-time">{item.createdAt}</span>
              </div>

              <div className="notification-item-sub">
                <span className="notification-path">
                  영향 받은 파일: {item.affectedFilesCount}개
                  {item.affectedPaths && item.affectedPaths.length > 0 && (
                    <span style={{ marginLeft: 8, fontSize: 12, color: '#9ca3af' }}>
                      ({item.affectedPaths.slice(0, 3).join(', ')}
                      {item.affectedPaths.length > 3 ? '...' : ''})
                    </span>
                  )}
                </span>
                {item.aiScore != null && (
                  <span className="notification-detail-preview" style={{ marginLeft: 8 }}>
                    AI 점수: {(item.aiScore * 100).toFixed(1)}%
                  </span>
                )}
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default NotificationPage;
