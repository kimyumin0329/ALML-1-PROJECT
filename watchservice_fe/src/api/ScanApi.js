// src/api/ScanApi.js
import { get, post } from './HttpClient';

// 명세: POST /scan/start  Body: { paths:[...], autoStartWatcher:true/false } -> {scanId}
export function startScan(paths = [], autoStartWatcher = true) {
  return post('/scan/start', { paths, autoStartWatcher });
}

export function pauseScan(scanId) {
  return post(`/scan/${scanId}/pause`);
}

export function fetchScanProgress(scanId) {
  return get(`/scan/${scanId}/progress`);
}
