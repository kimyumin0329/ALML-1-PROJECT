/**
 * 파일 이름 : SettingApi.js
 * 기능 : 설정 관련 API 함수를 제공한다. 감시 폴더, 예외 규칙, 알림 설정, 리셋, 피드백 등의 API를 포함한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
const API_BASE = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080';

/**
 * 함수 이름 : request
 * 기능 : HTTP 요청을 보내고 응답을 처리한다.
 * 매개변수 : path - API 경로, options - 요청 옵션 (method, body)
 * 반환값 : Promise - 응답 데이터
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
async function request(path, { method = 'GET', body } = {}) {
  const url = `${API_BASE}${path}`;

  const res = await fetch(url, {
    method,
    headers: {
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });

  // 204 No Content
  if (res.status === 204) return null;

  const text = await res.text();
  const data = text
    ? (() => {
        try {
          return JSON.parse(text);
        } catch {
          return text;
        }
      })()
    : null;

  if (!res.ok) {
    const msg =
      (data && typeof data === 'object' && (data.message || data.error)) ||
      (typeof data === 'string' ? data : '') ||
      `HTTP ${res.status}`;
    throw new Error(msg);
  }

  return data;
}

/** =========================
 * Watched Folders
 * ========================= */
export async function fetchWatchedFolders() {
  return request('/settings/folders', { method: 'GET' });
}

// ✅ 폴더 선택 다이얼로그 호출(백엔드 Swing)
export async function pickFolderPath() {
  return request('/settings/folders/pick', { method: 'GET' });
}

// ✅ useWatchedFolders.js에서 쓰는 이름으로 별칭 제공 (둘 다 동작)
export async function pickWatchedFolderPath() {
  return pickFolderPath();
}

export async function createWatchedFolder(payload) {
  // payload: {name, path}
  return request('/settings/folders', { method: 'POST', body: payload });
}

export async function deleteWatchedFolder(id) {
  return request(`/settings/folders/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

/** =========================
 * Exception Rules
 * ========================= */
export async function fetchExceptionRules() {
  return request('/settings/exceptions', { method: 'GET' });
}

export async function createExceptionRule(payload) {
  // payload: {type, pattern, memo}
  return request('/settings/exceptions', { method: 'POST', body: payload });
}

export async function deleteExceptionRule(id) {
  return request(`/settings/exceptions/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

/** =========================
 * Notification Settings
 * (트레이 빼기로 했으니 백엔드 미구현이어도 함수는 유지 가능)
 * ========================= */
export async function fetchNotificationSettings() {
  return request('/settings/notification', { method: 'GET' });
}

export async function updateNotificationSettings(payload) {
  return request('/settings/notification', { method: 'PUT', body: payload });
}

/** =========================
 * Reset
 * ========================= */
export async function resetSettings() {
  return request('/settings/reset', { method: 'POST', body: {} });
}

/** =========================
 * Feedback
 * ========================= */
export async function sendFeedback(payload) {
  return request('/support/feedback', { method: 'POST', body: payload });
}
