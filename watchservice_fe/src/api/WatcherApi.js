/**
 * 파일 이름 : WatcherApi.js
 * 기능 : 파일 감시 시작/중지 API 함수를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
import { post } from './HttpClient';

/**
 * 함수 이름 : startWatcher
 * 기능 : 지정된 폴더에 대한 파일 감시를 시작한다.
 * 매개변수 : folderPath - 감시할 폴더 경로
 * 반환값 : Promise - 시작 결과
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function startWatcher(folderPath) {
  const encoded = encodeURIComponent(folderPath);
  // 백엔드는 @PostMapping("/start") 로 받으니까 POST로 호출
  return post(`/watcher/start?folderPath=${encoded}`);
}

/**
 * 함수 이름 : stopWatcher
 * 기능 : 파일 감시를 중지한다.
 * 매개변수 : 없음
 * 반환값 : Promise - 중지 결과
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function stopWatcher() {
  return post('/watcher/stop');
}
