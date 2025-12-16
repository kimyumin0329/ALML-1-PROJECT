// src/pages/mainboard/MainBoardPage.jsx
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { startWatcher, stopWatcher } from '../../api/WatcherApi';
import { useWatchedFolders } from '../../hooks/UseWatchedFolders';
import { useLogs } from '../../hooks/UseLogs';
import { fetchDashboardSummary } from '../../api/DashboardApi';
import { startScan, pauseScan, fetchScanProgress } from '../../api/ScanApi';

import ProtectionStatusBadge from '../../components/protection/ProtectionStatusBadge';
import ScanControlPanel from '../../components/protection/ScanControlPanel';
import RecentEventsPanel from '../../components/protection/RecentEventsPanel';

function MainBoardPage() {
  // 대시보드 요약
  const [protectionStatus, setProtectionStatus] = useState('안전');
  const [statusCode, setStatusCode] = useState('SAFE');
  const [lastEventTime, setLastEventTime] = useState('N/A');
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState(null);

  // 감시/검사 상태
  const [isWatching, setIsWatching] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [scanProgress, setScanProgress] = useState(0);
  const [scanId, setScanId] = useState(null);

  const pollRef = useRef(null);

  // 폴더
  const { folders, promptAndAddFolder, removeFolder } = useWatchedFolders();

  // 최근 이벤트(5건)
  const { logs: recentLogs, loading: logsLoading, error: logsError } = useLogs(5);

  // 요약 불러오기
  useEffect(() => {
    const loadSummary = async () => {
      try {
        setSummaryLoading(true);
        setSummaryError(null);

        const data = await fetchDashboardSummary();
        if (!data) return;

        setProtectionStatus(data.statusLabel || '안전');
        setStatusCode(data.status || 'SAFE');
        setLastEventTime(data.lastEventTime || 'N/A');
      } catch (e) {
        console.error('fetchDashboardSummary error:', e);
        setSummaryError(e);
      } finally {
        setSummaryLoading(false);
      }
    };

    loadSummary();
  }, []);

  // 로그 -> 최근 이벤트 가공
  const recentEvents = useMemo(() => {
    const mapLogToEventView = (log) => {
      let level = 'info';
      if (log.aiLabel === 'DANGER') level = 'danger';
      else if (log.aiLabel === 'WARNING') level = 'warning';

      const message =
        log.aiDetail && log.aiDetail.length > 40
          ? log.aiDetail.slice(0, 40) + '...'
          : log.aiDetail || log.eventType || '파일 이벤트';

      return {
        id: log.id,
        time: log.collectedAt,
        path: log.path,
        level,
        message,
      };
    };

    return Array.isArray(recentLogs) ? recentLogs.map(mapLogToEventView) : [];
  }, [recentLogs]);

  // 감시 시작/중지
  const handleToggleWatch = async () => {
    const target = folders[0];

    if (!isWatching) {
      if (!target?.path) {
        alert('감시할 폴더가 없습니다.\n먼저 "폴더 추가"에서 경로를 등록해주세요.');
        return;
      }
      try {
        await startWatcher(target.path);
        setIsWatching(true);
      } catch (e) {
        console.error(e);
        alert('감시 시작 중 오류가 발생했습니다.\n' + e.message);
      }
      return;
    }

    try {
      await stopWatcher();
      setIsWatching(false);
    } catch (e) {
      console.error(e);
      alert('감시 중지 중 오류가 발생했습니다.\n' + e.message);
    }
  };

  // 즉시 검사
  const handleScanNow = async () => {
    const paths = folders.map((f) => f.path).filter(Boolean);
    if (paths.length === 0) {
      alert('검사할 폴더가 없습니다.\n먼저 "폴더 추가"에서 경로를 등록해주세요.');
      return;
    }

    // 이전 스캔 흔적 정리
    setScanId(null);
    setIsScanning(true);
    setScanProgress(0);

    try {
      // ✅ autoStartWatcher=true로 명시(백엔드 기본도 true지만 명확히)
      const res = await startScan(paths, true);
      const id = res?.scanId || res?.id || null;

      if (id) {
        setScanId(id);
        // 백엔드가 scan 완료 후 watcher 자동 시작하므로 UI도 감시중 표시(낙관적)
        setIsWatching(true);
        return;
      }

      // scanId 못받으면 실패로 처리
      throw new Error('scanId not returned');
    } catch (e) {
      // 백엔드 scan 미구현/실패 시 watcher로 fallback (옵션)
      console.warn('startScan failed, fallback startWatcher:', e);
      try {
        await startWatcher(paths[0]);
        setIsWatching(true);
      } catch (e2) {
        console.error(e2);
        alert('즉시 검사 시작 중 오류가 발생했습니다.\n' + e2.message);
      } finally {
        setIsScanning(false);
        setScanProgress(0);
      }
    }
  };

  // 검사 중지(=pause endpoint지만 실질 stop)
  const handlePause = async () => {
    try {
      if (scanId) {
        await pauseScan(scanId);

        // ✅ pause는 scan job 종료(PAUSED)로 처리하므로 polling도 정리
        setScanId(null);
        setScanProgress(0);

        if (pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } else {
        // scanId가 없으면 watcher 중지로 fallback
        await stopWatcher();
        setIsWatching(false);
      }
    } catch (e) {
      console.error(e);
      alert('검사 중지 중 오류가 발생했습니다.\n' + e.message);
    } finally {
      setIsScanning(false);
    }
  };

  // scan progress polling (scanId 있을 때만)
  useEffect(() => {
    if (!scanId) return;

    if (pollRef.current) clearInterval(pollRef.current);

    pollRef.current = setInterval(async () => {
      try {
        const p = await fetchScanProgress(scanId);
        const percent = Number(p?.percent ?? 0);
        if (Number.isFinite(percent)) setScanProgress(percent);

        // DONE 처리
        if (p?.status === 'DONE' || percent >= 100) {
          setIsScanning(false);
          setScanProgress(100);
          setScanId(null); // ✅ 완료 후 scanId 정리

          clearInterval(pollRef.current);
          pollRef.current = null;
          return;
        }

        // PAUSED/ERROR 처리
        if (p?.status === 'PAUSED' || p?.status === 'ERROR') {
          setIsScanning(false);
          setScanId(null);

          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch (e) {
        console.warn('fetchScanProgress failed:', e);
        setIsScanning(false);
        setScanId(null);

        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    }, 800);

    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, [scanId]);

  // 폴더 추가/삭제
  const handleAddFolder = () => promptAndAddFolder();
  const handleRemoveFolder = (id) => removeFolder(id);

  return (
    <div className="mainboard-root">
      <header className="mainboard-header">
        <h1>WatchService Agent</h1>
      </header>

      <div className="mainboard-body">
        <section className="main-left">
          <ProtectionStatusBadge
            protectionStatus={protectionStatus}
            statusCode={statusCode}
            lastEventTime={lastEventTime}
            summaryLoading={summaryLoading}
            summaryError={summaryError}
          />

          <div className="status-panel" style={{ marginTop: 12 }}>
            <div className="panel-header">
              <h2>검사 / 감시 제어</h2>
            </div>

            <div className="status-body">
              <div className="status-left" />
              <ScanControlPanel
                isWatching={isWatching}
                isScanning={isScanning}
                scanProgress={scanProgress}
                onToggleWatch={handleToggleWatch}
                onScanNow={handleScanNow}
                onPause={handlePause}
              />
            </div>
          </div>

          <div className="folder-panel">
            <div className="panel-header">
              <h2>감시 대상 폴더</h2>
            </div>

            <div className="folder-list">
              {folders.map((f) => (
                <div key={f.id} className="folder-item">
                  <div className="folder-name">
                    {f.name}
                    <div style={{ fontSize: '11px', color: '#9ca3af', marginTop: '2px' }}>
                      {f.path}
                    </div>
                  </div>
                  <button className="btn-icon" onClick={() => handleRemoveFolder(f.id)}>
                    삭제
                  </button>
                </div>
              ))}

              {folders.length === 0 && (
                <p style={{ fontSize: 13, color: '#9ca3af' }}>
                  아직 등록된 감시 폴더가 없습니다.
                  <br />
                  아래 버튼으로 먼저 폴더를 추가해 주세요.
                </p>
              )}
            </div>

            <button className="btn btn-outline" onClick={handleAddFolder}>
              폴더 추가
            </button>
          </div>
        </section>

        <section className="main-right">
          <RecentEventsPanel events={recentEvents} loading={logsLoading} error={logsError} />
        </section>
      </div>
    </div>
  );
}

export default MainBoardPage;
