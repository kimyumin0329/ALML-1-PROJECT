# WatchService Agent (Backend)

파일 시스템 이벤트(생성/수정/삭제/이름변경)를 감시하고, 이벤트를 분석(엔트로피/해시/스냅샷 비교)하여 위험도를 산출한 뒤 로그/알림/대시보드로 제공하는 Spring Boot 백엔드입니다.

## 1) 주요 기능

- **폴더 감시(Watcher)**: 지정 폴더의 파일 이벤트 감지(생성/수정/삭제/이름 변경)
- **파일 분석(Collector)**: 파일 크기 변화, 엔트로피 변화(암호화 의심), 해시 계산, 확장자 변화 감지
- **윈도우 집계(Analytics)**: 일정 시간창 단위로 이벤트를 묶어 특징값(feature) 생성
- **AI 연동(AI)**: 특징값을 외부 AI 서버로 전송해 위험 레이블/점수 수신
- **로그 저장(Storage)**: 분석 결과를 로그로 저장/조회/삭제/내보내기
- **알림(Alerts)**: 위험/경고 이벤트를 알림으로 제공 + 통계 조회
- **대시보드(Dashboard)**: 보호 상태 요약(녹/노/적), 최근 이벤트, 위험 카운트 제공
- **설정(Settings)**: 감시 폴더/예외 규칙 관리
- **수동 스캔(Scan)**: 즉시 검사(스캔) 시작/일시정지/진행률 조회

---

## 2) 아키텍처 다이어그램

아래 다이어그램은 “파일 이벤트 발생 → 분석/집계 → AI 판단 → 저장/표시” 흐름을 보여줍니다.

```mermaid
flowchart LR
    subgraph OS[OS / File System]
      FS[(Files & Folders)]
    end

    subgraph W[watcher]
      WS[WatcherService]
      WC[WatcherController]
      EVT[WatcherEvent / WatcherEventRecord]
    end

    subgraph C[collector]
      FCS[FileCollectorService]
      EA[EntropyAnalyzer]
      HC[HashCalculator]
      SS[FileSnapshotStore]
      FAR[FileAnalysisResult]
    end

    subgraph A[analytics]
      EWA[EventWindowAggregator]
      PAY[AiPayload (features)]
    end

    subgraph AI[ai]
      ACS[AiService]
      EXT[(External AI Server\n/api/analyze)]
      RES[AiResult]
    end

    subgraph S[storage]
      LS[LogService]
      LR[LogRepository]
      DB[(SQLite log.db)]
    end

    subgraph UI[Read APIs]
      DC[DashboardController\n/dashboard/summary]
      AC[AlertController\n/alerts/*]
      LC[LogController\n/logs/*]
      SC[SettingsController\n/settings/*]
    end

    FS --> WS
    WC --> WS
    WS --> EVT --> FCS
    FCS --> EA --> FAR
    FCS --> HC --> FAR
    SS <--> FCS
    FAR --> EWA --> PAY
    PAY --> ACS --> EXT --> ACS --> RES
    RES --> LS --> LR --> DB

    DB --> DC
    DB --> AC
    DB --> LC
    DB --> SC
```

> Mermaid 다이어그램은 GitHub/Notion 등에서 렌더링될 수 있습니다. (렌더링이 안 되면 코드 블록 그대로 다이어그램 소스로 사용 가능)

---

## 3) 기술 스택

- Java 17
- Spring Boot 3.x
- (저장소) SQLite 기반 로그/설정 저장
- (외부 연동) HTTP 기반 AI 서버 호출

---

## 4) 프로젝트 구조(요약)

```text
watchservice_agent/
├── watcher/     # 감시 시작/중지, 이벤트 발생
├── collector/   # 파일 단위 분석(엔트로피/해시/스냅샷 비교)
├── analytics/   # 시간창(window) 집계, feature 생성
├── ai/          # 외부 AI 서버 호출, 결과 변환
├── storage/     # 로그 저장/조회/삭제/내보내기
├── alerts/      # 알림 목록/통계 제공
├── dashboard/   # 메인 보드 요약 제공
├── settings/    # 감시 폴더/예외 규칙 관리
├── scan/        # 수동 스캔 기능
└── common/      # 공통 설정(CORS 등), 유틸
```

---

## 5) 실행 방법

### 5.1 요구 사항
- JDK 17+
- Gradle(또는 Gradle Wrapper)

### 5.2 실행
```bash
./gradlew bootRun
```

기본 포트:
- `http://localhost:8080`

---

## 6) 외부 AI 서버 연동

백엔드는 특징값(`AiPayload`)을 외부 AI 서버로 전송해 위험 판단 결과를 받아옵니다.

### 6.1 설정(application.yml)
```yaml
ai:
  analyze:
    url: http://localhost:8001/api/analyze
  family:
    url: http://localhost:8001/predict
```

### 6.2 요청 JSON 형식 (AiPayload)
`POST /ai/analyze`에 들어오는 body 형식(동일하게 외부 AI 서버로 전달됨):

```json
{
  "file_read_count": 2,
  "file_write_count": 15,
  "file_delete_count": 0,
  "file_rename_count": 3,
  "file_encrypt_like_count": 2,
  "changed_files_count": 7,
  "random_extension_flag": 1,
  "entropy_diff_mean": 0.41,
  "file_size_diff_mean": 12345.0
}
```

- `*_count`: 이벤트 수(집계 구간 내)
- `random_extension_flag`: 랜덤 확장자 의심이면 1
- `entropy_diff_mean`: 엔트로피 변화 평균(암호화 의심 지표)
- `file_size_diff_mean`: 파일 크기 변화 평균 (바이트)

---

## 7) API 요약

### 7.1 Dashboard
- `GET /dashboard/summary`  
  메인보드 요약(상태, 마지막 이벤트 시간, 위험/경고 카운트 등)

### 7.2 Watcher
- `POST /watcher/start`
- `POST /watcher/stop`

### 7.3 Scan
- `POST /scan/start`
- `POST /scan/{scanId}/pause`
- `GET  /scan/{scanId}/progress`

### 7.4 Logs (Storage)
- `GET    /logs`
- `GET    /logs/recent`
- `GET    /logs/{id}`
- `DELETE /logs/{id}`
- `POST   /logs/delete`
- `POST   /logs/export`

### 7.5 Settings
- `GET/POST/DELETE /settings/folders`
- `GET/POST/DELETE /settings/exceptions`

### 7.6 Alerts
- `GET /alerts`
- `GET /alerts/{id}`
- `GET /alerts/stats`

### 7.7 AI
- `POST /ai/analyze`
- `GET  /ai/ping`
- (추가 시) `POST /ai/family/predict`

---

## 8) 테스트 예시

### 8.1 AI 분석 테스트
```bash
curl -X POST "http://localhost:8080/ai/analyze"   -H "Content-Type: application/json"   -d '{
    "file_read_count": 2,
    "file_write_count": 15,
    "file_delete_count": 0,
    "file_rename_count": 3,
    "file_encrypt_like_count": 2,
    "changed_files_count": 7,
    "random_extension_flag": 1,
    "entropy_diff_mean": 0.41,
    "file_size_diff_mean": 12345.0
  }'
```

---

## 9) 문제 해결(Troubleshooting)

### 9.1 AI 서버 호출 실패(Connection refused)
- `ai.analyze.url`에 설정된 포트에 AI 서버가 실제로 떠 있는지 확인
```bash
lsof -iTCP:8001 -sTCP:LISTEN -nP
```

### 9.2 AI 서버 404 Not Found
- AI 서버에 `/api/analyze` 라우트가 존재하는지 확인
- FastAPI라면 `http://localhost:8001/docs`에서 라우트 목록 확인

---

## 10) License
내부 프로젝트/캡스톤 용도 (필요 시 작성)
