# WatchService Agent Backend 구조 및 역할 정리 (information.md)

아래 내용은 `watchservice_agent` 백엔드의 폴더/파일 구조를 기준으로, 각 모듈과 주요 파일이 **무슨 역할을 하는지**를 요약한 문서입니다.

---

## 1) `WatchserviceAgentApplication.java`

- 스프링부트 앱 엔트리 포인트(실행 시작점)

---

## 2) `watcher/` (파일 감시 시작/중지 + 이벤트 발생)

### `WatcherController.java`
- 감시 시작/중지 API 제공
- (대략) `/watcher/start`, `/watcher/stop` 형태로 호출됨

### `WatcherService.java`
- 실제 파일 감시 로직(WatchService 구동/종료, 이벤트 수집 트리거)

### `WatcherRepository.java`
- 감시 관련 상태/설정 조회용
- (프로젝트 방식에 따라 상태 저장/조회 역할)

### `domain/WatcherEvent.java`
- 감시 이벤트(생성/수정/삭제 등) 도메인 모델

### `dto/WatcherEventRecord.java`
- 컨트롤러/서비스 간 이벤트 전달용 DTO(경로, 이벤트 타입, 시간 등)

---

## 3) `collector/` (이벤트 1건을 “분석 결과”로 바꾸는 곳)

### `FileCollectorService.java`
- `WatcherEventRecord` 1개를 받아서 파일 존재/크기/엔트로피/확장자 변화 등을 비교해 `FileAnalysisResult` 생성
- `FileSnapshotStore`에서 “이전 상태(last/baseline)”를 꺼내서 before/after 비교하는 구조

### `dto/FileAnalysisResult.java`
- 이벤트 1건에 대한 분석 결과(크기 전후, 엔트로피 전후, 확장자 전후, 변화량 등) 담는 DTO

### `business/EntropyAnalyzer.java`
- 파일 내용 샘플을 읽어 Shannon entropy 계산

### `business/HashCalculator.java`
- 파일 SHA-256 해시 계산 유틸(무결성/변조 판단용 재료)

### `snapshot/FileSnapshotStore.java`
- 파일별 스냅샷 저장소(메모리 맵 형태)
- baseline(처음 상태) / last(직전 상태)를 관리해서 비교 근거를 제공

### `snapshot/SnapshotConfig.java`
- `FileSnapshotStore`를 스프링 Bean으로 등록하는 설정

---

## 4) `analytics/` (여러 이벤트를 “윈도우 집계 특징값”으로)

### `EventWindowAggregator.java`
- 일정 시간(예: 최근 N초) 동안 들어온 `FileAnalysisResult`들을 모아서
    - `file_write_count`, `file_delete_count`, `file_rename_count`
    - `entropy_diff_mean`, `file_size_diff_mean`
    - `random_extension_flag`, `file_encrypt_like_count`
      같은 특징값(feature) 생성
- 생성한 특징값을 `AiPayload`로 만들어 `AiService` 호출 → 결과를 `LogService`로 저장하는 허브 역할

---

## 5) `ai/` (외부 AI 서버 연동)

### `AiController.java`
- “파이프라인 필수”라기보단 AI 연동 테스트용 엔드포인트
- 예: `POST /ai/analyze`, `GET /ai/ping`

### `AiService.java`
- 실제로 외부 AI 서버에 HTTP 요청 보내는 클라이언트
- `AiPayload`(요청) → `AiResponse`(응답) → `AiResult`(도메인 결과) 변환

### `dto/AiPayload.java`
- AI 서버로 보내는 입력 JSON 구조(집계 특징값들)

### `dto/AiResponse.java`
- AI 서버가 반환하는 원본 응답(JSON)을 받는 DTO

### `domain/AiResult.java`
- 백엔드 내부에서 쓰는 “최종 AI 판단 결과”(label/score/detail 등)

---

## 6) `storage/` (로그 저장/조회/내보내기)

### `LogController.java`
- 로그 조회/상세/삭제/내보내기 API 제공
- 예: `/logs`, `/logs/{id}`, `/logs/export`, `/logs/delete` 등

### `LogService.java`
- 로그 비즈니스 로직(필터/페이지네이션/내보내기 형식 변환/최근 로그/알림용 조회 등)

### `LogRepository.java`
- DB(SQLite 등)에 로그 CRUD 담당

### `LogWriterWorker.java`
- 로그를 “즉시”가 아니라 워커/큐 기반으로 비동기 저장하려는 역할(성능/버퍼링 목적)

### `domain/Log.java`
- 로그 레코드 도메인 모델

### `dto/*`
- `LogResponse`: 로그 1건 응답
- `LogPageResponse`: 페이지 형태 목록 응답
- `LogExportRequest`: 내보내기 요청(기간/형식 등)
- `LogDeleteRequest/Response`: 다건 삭제/응답

---

## 7) `alerts/` (알림 화면/통계용 “로그의 위험 이벤트” 뷰)

### `AlertController.java`
- `/alerts`, `/alerts/{id}`, `/alerts/stats` 같은 “알림용” API 제공
- 내부적으로는 `LogService`를 이용해 위험/경고 로그를 조회하는 구조
- (코드상 패키지명이 storage로 되어있는 부분이 보이는데, 기능은 alerts 폴더 역할임 — 리팩토링 포인트)

### `AlertService.java`
- 알림 관련 집계/조회 로직(프로젝트 설계상 분리된 서비스)

### `dto/AlertPageResponse / AlertStatsResponse / AlertStatusResponse`
- 알림 목록/통계/상태 요약 응답 DTO

---

## 8) `dashboard/` (메인보드 요약)

### `DashboardController.java`
- `GET /dashboard/summary` 같은 요약 API
- 최근 로그/위험 카운트/마지막 이벤트 시간/감시 경로 등을 조합해서 내려줌

### `dto/DashboardSummaryResponse.java`
- 대시보드에 필요한 요약 데이터 DTO

---

## 9) `settings/` (감시 폴더/예외 규칙 관리)

### `SettingsController.java`
- 감시 폴더 CRUD: `/settings/folders` GET/POST/DELETE
- 예외 규칙 CRUD: `/settings/exceptions` GET/POST/DELETE

### `SettingsService.java`
- 설정 비즈니스 로직(중복 방지, 유효성, 저장/조회)

### `SettingsRepository.java`
- 설정 저장소(DB 연동)

### `domain/WatchedFolder.java / ExceptionRule.java`
- 감시 폴더/예외규칙 도메인

### `dto/*`
- 요청/응답 DTO들

---

## 10) `scan/` (수동 “즉시 검사” 작업)

### `ScanController.java`
- `POST /scan/start` : 스캔 시작
- `POST /scan/{scanId}/pause` : 일시정지
- `GET /scan/{scanId}/progress` : 진행률 조회

### `ScanService.java`
- 스캔 잡 생성/진행 상태 업데이트/중단 처리

### `domain/ScanJob.java / ScanStatus.java`
- 스캔 작업과 상태(진행중/일시정지/완료 등)

### `dto/ScanStartRequest/Response, ScanProgressResponse, ScanPauseResponse`
- 스캔 API 요청/응답 구조

---

## 11) `common/` (공통 설정/유틸)

### `config/WebConfig.java`
- CORS 설정(React 개발 서버 `localhost:3000`, `5173` 허용)

### `util/SessionIdManager.java`
- 세션/실행 단위 식별자 관리(로그 묶기나 화면 세션 구분 등에 쓰기 좋은 유틸)
