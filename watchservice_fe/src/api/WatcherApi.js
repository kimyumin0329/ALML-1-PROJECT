// src/api/WatcherApi.js
import { post } from './HttpClient';

// 폴더 감시 시작
export function startWatcher(folderPath) {
  const encoded = encodeURIComponent(folderPath);
  // 백엔드는 @PostMapping("/start") 로 받으니까 POST로 호출
  return post(`/watcher/start?folderPath=${encoded}`);
}

// 감시 중지
export function stopWatcher() {
  return post('/watcher/stop');
}
