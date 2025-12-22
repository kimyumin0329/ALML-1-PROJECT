/**
 * 파일 이름 : HttpClient.js
 * 기능 : 백엔드 API와 통신하기 위한 HTTP 클라이언트 유틸리티. GET, POST, DELETE, PUT 메서드를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */

const BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8080';

/**
 * 함수 이름 : request
 * 기능 : HTTP 요청을 보내고 응답을 처리한다. JSON 응답은 자동으로 파싱한다.
 * 매개변수 : path - API 경로, options - fetch 옵션
 * 반환값 : Promise - JSON 객체 또는 텍스트
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
async function request(path, options = {}) {
  const url = `${BASE_URL}${path}`;

  const finalOptions = {
    ...options,
    headers: {
      ...(options.headers || {}),
    },
  };

  const res = await fetch(url, finalOptions);

  if (!res.ok) {
    const text = await res.text().catch(() => '');
    throw new Error(`HTTP ${res.status} ${res.statusText}${text ? ` - ${text}` : ''}`);
  }

  const ct = (res.headers.get('content-type') || '').toLowerCase();
  if (ct.includes('application/json')) return res.json();
  return res.text();
}

/**
 * 함수 이름 : get
 * 기능 : GET 요청을 보낸다.
 * 매개변수 : path - API 경로
 * 반환값 : Promise - 응답 데이터
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function get(path) {
  return request(path, { method: 'GET' });
}

/**
 * 함수 이름 : del
 * 기능 : DELETE 요청을 보낸다.
 * 매개변수 : path - API 경로
 * 반환값 : Promise - 응답 데이터
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function del(path) {
  return request(path, { method: 'DELETE' });
}

/**
 * 함수 이름 : post
 * 기능 : POST 요청을 보낸다. body가 없으면 쿼리스트링만 사용한다.
 * 매개변수 : path - API 경로, body - 요청 본문 (선택)
 * 반환값 : Promise - 응답 데이터
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function post(path, body) {
  // body 없으면 쿼리스트링만 사용하는 POST 같은 케이스 대응
  if (body === undefined) return request(path, { method: 'POST' });

  return request(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/**
 * 함수 이름 : put
 * 기능 : PUT 요청을 보낸다.
 * 매개변수 : path - API 경로, body - 요청 본문
 * 반환값 : Promise - 응답 데이터
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
export function put(path, body) {
  return request(path, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  });
}
