# Frontend 구조 및 파일 역할 (information.md)

아래는 `src/` 기준으로 폴더/파일 구조와 각 역할을 정리한 문서입니다.

---

## A. 앱 엔트리/라우팅 (src 루트)

- **`index.js`**: React 렌더 시작점. `<App />`를 DOM에 마운트.
- **`App.js`**: 라우터(Route) 정의.  
  `/`(메인보드), `/notifications`, `/logs`, `/settings/*` 페이지들을 연결.
- **`App.css`, `index.css`**: 전역 스타일.
- **`App.test.js`, `setupTests.js`, `reportWebVitals.js`**: CRA 기본 테스트/성능 측정용 보일러플레이트.
- **`logo.svg`**: 로고 리소스.

---

## B. API 레이어 (src/api) — 백엔드 엔드포인트 호출 래퍼

- **`HttpClient.js`**: 공통 `fetch` 래퍼
  - `BASE_URL = REACT_APP_API_BASE_URL || http://localhost:8080`
  - `get/post/put/del` 제공
  - `res.ok` 체크 + `JSON/text` 자동 파싱
- **`DashboardApi.js`**
  - `GET /dashboard/summary` : 대시보드 요약 조회
- **`LogsApi.js`**
  - `GET /logs/recent?limit=...` : 최근 로그
  - `GET /logs?...` : (필터/페이지용, 있으면 사용)
  - `GET /logs/{id}` : 단건 상세
  - `DELETE /logs/{id}` : 단건 삭제
  - `POST /logs/delete` : 선택 삭제
  - `POST /logs/export` : 내보내기
- **`NotificationsApi.js`**
  - `GET /alerts?page=1&size=...` : 알림 목록  
    (프론트 0-based → 백엔드 1-based 변환 포함)
  - `GET /alerts/{id}` : 알림 상세
  - `GET /alerts/stats?...` : 알림 통계
- **`ScanApi.js`**
  - `POST /scan/start` : 스캔 시작
  - `POST /scan/{scanId}/pause` : 일시정지
  - `GET /scan/{scanId}/progress` : 진행률 조회
- **`SettingApi.js`**
  - 감시폴더: `GET/POST/DELETE /settings/folders`, `POST /settings/folders/pick`
  - 예외규칙: `GET/POST/DELETE /settings/exceptions`
  - 알림설정: `GET/PUT /settings/notification`
  - 초기화: `POST /settings/reset`
  - 문의: `POST /support/feedback`
- **`WatcherApi.js`**
  - `POST /watcher/start?folderPath=...` : 감시 시작
  - `POST /watcher/stop` : 감시 중지

---

## C. Hooks (src/hooks) — 페이지에서 쓰는 상태/데이터 로딩 로직

- **`UseProtectionStatus.js`**
  - `fetchDashboardSummary()` 호출 → 보호상태 요약을 `summary`로 제공
- **`UseLogs.js`**
  - 로그 목록 상태/필터/페이지 관리
  - `/logs` 실패 시 `/logs/recent`로 fallback하는 로직 포함
- **`UseNotifications.js`**
  - 알림 목록/검색조건/페이지 상태 관리
  - `fetchAlerts / fetchAlertDetail / fetchAlertStats` 사용
- **`UseWatchedFolders.js`**
  - 감시폴더 목록 로드/추가/삭제
  - 폴더 선택(`pickFolderPath`) → 생성(`createWatchedFolder`) 흐름
- **`UseExceptions.js`**
  - 예외(화이트리스트) 규칙 목록/추가/삭제
  - `SettingExceptionsPage`에서 쓰기 좋게 별칭 제공

---

## D. 레이아웃 (src/layout)

- **`MainLayout.jsx`**
  - 상단 헤더 + 좌측 `NavSidebar` + 우측 콘텐츠(children) 레이아웃 골격

---

## E. Pages (src/pages) — Route에 매핑되는 화면 단위

- **`mainboard/MainBoardPage.jsx`**: 메인 보드(대시보드)
  - 보호상태 표시, 감시 시작/중지, 스캔 시작/일시정지/진행률,
    감시폴더 간단 관리, 최근 이벤트 표시
  - 사용 API/훅: `DashboardApi`, `WatcherApi`, `ScanApi`, `useLogs`, `useWatchedFolders`
- **`logs/LogsPage.jsx`**: 로그 조회/검색/상세/삭제/내보내기
  - `LogFilterBar`, `LogTable`, `LogDetailModal` 조합
  - 사용 API/훅: `LogsApi`, `useLogs`
- **`notifications/NotificationPage.jsx`**: 알림 목록 + 페이지네이션/필터
- **`notifications/NotificationDetailPage.jsx`**: 알림 상세  
  (라우팅 state가 없으면 서버에서 id로 재조회)
- **`notifications/NotificationStatsPage.jsx`**: 알림 통계 + 차트
- **`settings/SettingHomePage.jsx`**: 설정 홈(메뉴 카드)
- **`settings/SettingFoldersPage.jsx`**: 감시 폴더 관리 화면
- **`settings/SettingExceptionsPage.jsx`**: 예외 규칙 관리(패턴/type/memo 추가/삭제)
- **`settings/SettingNotifyPage.jsx`**: 알림 방식(팝업/소리)  
  서버 실패 시 `localStorage`로 fallback 하는 방어 로직 포함
- **`settings/SettingResetPage.jsx`**: 설정 초기화 + 로컬 캐시 제거
- **`settings/SettingUpdatePage.jsx`**: 업데이트 안내(현재는 알림만 띄우는 플레이스홀더)
- **`settings/SettingFeedbackPage.jsx`**: 문의/피드백 전송(`/support/feedback`)

---

## F. Components (src/components) — 재사용 UI 블록

### 1) common
- **`NavSidebar.jsx`**: 좌측 네비게이션(메인/알림/로그/설정 링크)
- **`Button.jsx`, `HeaderBar.jsx`, `Modal.jsx`, `ProgressBar.jsx`, `Table.jsx`**: 0바이트(빈 파일)  
  → 공통 컴포넌트 “자리”만 잡아둔 상태라 실제 구현이 필요하거나, 안 쓰면 삭제해도 됨.

### 2) protection (메인보드)
- **`ProtectionStatusBadge.jsx`**: SAFE/WARNING/DANGER 색상 점 + 상태 텍스트 표시
- **`ScanControlPanel.jsx`**: 감시 토글/즉시검사/일시정지 버튼 + 진행률 바
- **`RecentEventsPanel.jsx`**: 최근 이벤트 리스트(라벨에 따라 danger/warning/info 뱃지)

### 3) logs
- **`LogFilterBar.jsx`**: 키워드/기간/정렬/라벨 등 필터 UI + 검색/초기화/새로고침
- **`LogTable.jsx`**: 로그 테이블 + 체크박스(전체/개별 선택)
- **`LogDetailModal.jsx`**: 단건 로그 상세 모달

### 4) folders
- **`FolderListManager.jsx`**: 감시폴더 목록 렌더링 + 추가/삭제 버튼

### 5) notifications
- **`NotificationList.jsx`**: 알림 리스트 컨테이너(로딩/에러/빈 상태 처리)
- **`NotificationItem.jsx`**: 알림 1개 행(레벨 표시/요약/클릭 처리)
- **`NotificationDetailView.jsx`**: 상세 표시용 뷰 컴포넌트(페이지에서 재사용 가능)
- **`NotificationStatusChart.jsx`**: 통계 시각화(간단 바/비율 표시)
