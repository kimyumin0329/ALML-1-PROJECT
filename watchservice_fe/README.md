# WatchService Agent Frontend (React)

파일 감시/랜섬웨어 의심 탐지 시스템(WatchService Agent)의 **프론트엔드(React SPA)** 입니다.  
메인 보드(보호 상태), 알림(히스토리/상세/통계), 로그 관리(조회/상세/검색/내보내기/삭제), 설정(감시폴더/예외/알림/초기화/업데이트/문의) 화면을 제공합니다.

> 백엔드(Spring Boot)와 REST API로 통신하며, 기본 API 주소는 `http://localhost:8080` 입니다.

---

## 주요 기능

- **메인 보드**
  - 보호 상태(예: SAFE/WARNING/DANGER) 요약 표시
  - 감시 시작/중지
  - 즉시 검사(스캔) 시작/일시정지/진행률 조회
  - 감시 폴더 간단 관리(목록/추가/삭제)
  - 최근 이벤트(로그) 표시

- **알림**
  - 알림 목록/검색/페이지네이션
  - 알림 상세
  - 알림 통계(차트)

- **로그**
  - 로그 목록/필터/검색
  - 상세 모달
  - 선택 삭제/전체 삭제
  - 내보내기(export)

- **설정**
  - 감시 폴더 관리
  - 예외(화이트리스트) 규칙 관리
  - 알림 방식 설정
  - 초기화(설정 리셋)
  - 업데이트 안내(플레이스홀더)
  - 문의/피드백 전송

---

## 기술 스택

- React (SPA)
- React Router (라우팅)
- Fetch 기반 HTTP 래퍼 (`src/api/HttpClient.js`)
- CSS (전역/컴포넌트/레이아웃)

---

## 빠른 시작

### 1) 설치
```bash
npm install
```

### 2) 실행
```bash
npm start
```

기본적으로 개발 서버는 `http://localhost:3000` 에서 열립니다.

---

## 환경 변수

### `REACT_APP_API_BASE_URL`
백엔드 API 베이스 URL을 지정합니다.

- 기본값: `http://localhost:8080`

예시:
```bash
# macOS / Linux
export REACT_APP_API_BASE_URL="http://localhost:8080"

# Windows PowerShell
setx REACT_APP_API_BASE_URL "http://localhost:8080"
```

---

## 폴더 구조

```text
src/
├── api/                      # 백엔드 API 호출 래퍼
│   ├── HttpClient.js         # 공통 HTTP 클라이언트 (BASE_URL/에러처리/JSON 파싱)
│   ├── DashboardApi.js       # /dashboard/summary
│   ├── LogsApi.js            # /logs (조회/상세/삭제/내보내기)
│   ├── NotificationsApi.js   # /alerts (목록/상세/통계)
│   ├── ScanApi.js            # /scan (start/pause/progress)
│   ├── SettingApi.js         # /settings + /support/feedback
│   └── WatcherApi.js         # /watcher (start/stop)
├── hooks/                    # 데이터 로딩/상태 관리 커스텀 훅
├── components/               # 재사용 UI 컴포넌트
│   ├── protection/           # 메인보드(보호/스캔/최근 이벤트)
│   ├── notifications/        # 알림(리스트/아이템/상세/차트)
│   ├── logs/                 # 로그(필터/테이블/상세모달)
│   ├── folders/              # 감시 폴더 리스트/관리
│   └── common/               # 공통 컴포넌트 자리(일부 파일은 빈 파일)
├── pages/                    # 라우팅 단위 화면
├── layout/                   # 공통 레이아웃(MainLayout)
├── styles/                   # CSS 분리(컴포넌트/레이아웃)
├── App.js                    # 라우팅 구성
└── index.js                  # 엔트리 포인트
```

---

## 화면(라우트) 구성

`src/App.js`에서 라우트를 정의합니다(예: `/`, `/logs`, `/notifications`, `/settings/*`).

- `/` : MainBoardPage (메인 보드)
- `/notifications` : NotificationPage (알림 목록)
- `/notifications/:id` : NotificationDetailPage (알림 상세)
- `/notifications/stats` : NotificationStatsPage (알림 통계)
- `/logs` : LogsPage (로그 관리)
- `/settings` : SettingHomePage (설정 홈)
- `/settings/folders` : SettingFoldersPage (감시폴더)
- `/settings/exceptions` : SettingExceptionsPage (예외 규칙)
- `/settings/notify` : SettingNotifyPage (알림 방식)
- `/settings/reset` : SettingResetPage (초기화)
- `/settings/update` : SettingUpdatePage (업데이트)
- `/settings/feedback` : SettingFeedbackPage (문의)

---

## 동작 흐름(간단 다이어그램)

```mermaid
flowchart LR
  UI[React Pages/Components] --> Hooks[Custom Hooks]
  Hooks --> API[API Layer]
  API --> HTTP[HttpClient (fetch)]
  HTTP --> BE[Spring Boot Backend]
  BE --> UI
```

---

## API 엔드포인트 요약

### Dashboard
- `GET /dashboard/summary` : 보호 상태 요약

### Watcher
- `POST /watcher/start?folderPath=...` : 감시 시작
- `POST /watcher/stop` : 감시 중지

### Scan
- `POST /scan/start` : 스캔 시작
- `POST /scan/{scanId}/pause` : 스캔 일시정지
- `GET /scan/{scanId}/progress` : 진행률 조회

### Logs
- `GET /logs/recent?limit=...` : 최근 로그
- `GET /logs` : 로그 조회(필터/검색)
- `GET /logs/{id}` : 로그 상세
- `DELETE /logs/{id}` : 로그 삭제(단건)
- `POST /logs/delete` : 로그 삭제(선택)
- `POST /logs/export` : 로그 내보내기

### Alerts (Notifications)
- `GET /alerts?page=1&size=...` : 알림 목록  
  - 프론트는 0-based 페이지를 사용하고, API 호출 시 1-based로 변환하는 로직이 포함되어 있습니다.
- `GET /alerts/{id}` : 알림 상세
- `GET /alerts/stats` : 알림 통계

### Settings / Support
- `GET /settings/folders` : 감시폴더 목록
- `POST /settings/folders` : 감시폴더 추가
- `DELETE /settings/folders/{id}` : 감시폴더 삭제
- `POST /settings/folders/pick` : OS 폴더 선택(다이얼로그) 요청(백엔드 지원 필요)
- `GET /settings/exceptions` : 예외 규칙 목록
- `POST /settings/exceptions` : 예외 규칙 추가
- `DELETE /settings/exceptions/{id}` : 예외 규칙 삭제
- `GET /settings/notification` : 알림 설정 조회
- `PUT /settings/notification` : 알림 설정 저장
- `POST /settings/reset` : 설정 초기화
- `POST /support/feedback` : 문의/피드백 전송

---

## 개발/유지보수 메모

### `components/common`의 빈 파일들
아래 파일들은 현재 **0바이트(빈 파일)** 로 확인됩니다.  
공통 UI 컴포넌트로 사용할 계획이면 구현이 필요하고, 사용하지 않으면 삭제해도 됩니다.

- `src/components/common/Button.jsx`
- `src/components/common/HeaderBar.jsx`
- `src/components/common/Modal.jsx`
- `src/components/common/ProgressBar.jsx`
- `src/components/common/Table.jsx`

### 실패 시 fallback 로직(예시)
- 로그 조회에서 `/logs` 실패 시 `/logs/recent`로 대체 조회하는 흐름이 포함되어 있습니다.
- 알림 방식 설정은 서버 실패 시 `localStorage`를 사용하는 방어 로직이 포함될 수 있습니다.

---

## 스크립트

- `npm start` : 개발 서버 실행
- `npm test` : 테스트 실행
- `npm run build` : 프로덕션 빌드

---

## 라이선스
프로젝트 내부 정책에 따릅니다.
