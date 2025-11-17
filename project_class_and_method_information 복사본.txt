- watcher  
| 파일                          | 클래스 / 메서드                                        | 설명                         |
| --------------------------- | ------------------------------------------------ | -------------------------- |
| **WatcherEvent.java**       | `WatcherEvent`                                   | 파일 이벤트(생성/수정/삭제) 도메인 객체    |
|                             | `boolean isDirectory()`                          | 경로가 폴더인지 판별                |
|                             | `boolean isFile()`                               | 경로가 파일인지 판별                |
| **WatcherPath.java**        | `WatcherPath`                                    | 감시 대상 경로를 표현하는 모델          |
|                             | `boolean exists()`                               | 경로 존재 여부 확인                |
|                             | `boolean isValid()`                              | 폴더 유효성 검사                  |
| **WatcherEventRecord.java** | `WatcherEventRecord`                             | Collector로 전달되는 감시 이벤트 DTO |
| **WatcherService.java**     | `startWatching(String folderPath)`               | 지정 경로 감시 시작                |
|                             | `stopWatching()`                                 | 감시 중지 및 자원 해제              |
|                             | `processEvent(Path path, String eventType)`      | 감지된 이벤트를 Collector로 전달     |
| **WatcherController.java**  | `startWatching(@RequestParam String folderPath)` | 감시 시작 API                  |
|                             | `stopWatching()`                                 | 감시 중지 API                  |
| **WatcherRepository.java**  | `savePath(String folderPath)`                    | 감시 경로 로컬 저장                |
|                             | `getLastWatchedPath()`                           | 최근 감시 경로 조회                |

- collector  
| 파일                            | 클래스 / 메서드                           | 설명                                   |
| ----------------------------- | ----------------------------------- | ------------------------------------ |
| **EntropyAnalyzer.java**      | `EntropyAnalyzer`                   | Shannon Entropy 계산 담당                |
|                               | `calculateEntropy(String filePath)` | 파일의 무작위도 계산                          |
| **HashCalculator.java**       | `HashCalculator`                    | 파일 해시 계산 담당                          |
|                               | `calculateSHA256(String filePath)`  | SHA-256 해시 계산                        |
|                               | `calculateMD5(String filePath)`     | MD5 해시 계산                            |
| **FileData.java**             | `FileData`                          | 파일의 메타데이터 모델                         |
|                               | `fromPath(String filePath)`         | 파일 크기·확장자·수정시각 수집                    |
| **FileAnalysisResult.java**   | `FileAnalysisResult`                | Collector 결과 DTO                     |
| **FileCollectorService.java** | `collect(String filePath)`          | 파일 분석 총괄 (Hash + Entropy 계산 후 결과 반환) |

- storage  
| 파일                     | 클래스 / 메서드                         | 설명                   |
| ---------------------- | --------------------------------- | -------------------- |
| **Log.java**           | `Log`                             | DB에 저장되는 로그 엔티티      |
| **LogRepository.java** | `save(Log log)`                   | 로그 저장                |
|                        | `findAll()`                       | 전체 로그 조회             |
|                        | `deleteById(Long id)`             | ID로 로그 삭제            |
| **LogService.java**    | `save(FileAnalysisResult result)` | Collector 결과를 DB에 저장 |
|                        | `getAllLogs()`                    | 모든 로그 조회             |
|                        | `deleteLog(Long id)`              | 로그 삭제                |
| **LogController.java** | `getAllLogs()`                    | REST API — 로그 조회     |
|                        | `deleteLog(Long id)`              | REST API — 로그 삭제     |

- ai  
| 파일                    | 클래스 / 메서드                              | 설명                               |
| --------------------- | -------------------------------------- | -------------------------------- |
| **AiController.java** | `analyze(AiPayload payload)`           | Collector 결과를 AI 서버에 전달          |
| **AiService.java**    | `requestAnalysis(AiPayload payload)`   | AI 서버로 POST 요청 전송                |
|                       | `convertToDomain(AiResponse response)` | AI 응답을 도메인 객체로 변환                |
| **AiPayload.java**    | `AiPayload`                            | Collector → AI 서버 요청 DTO         |
| **AiResponse.java**   | `AiResponse`                           | AI 서버 응답 DTO                     |
| **AiResult.java**     | `AiResult`                             | AI 분석 결과 도메인 모델 (위험 점수, 판정 결과 등) |

- common
| 폴더             | 역할                              |
| -------------- | ------------------------------- |
| **config/**    | Spring 설정, Bean 등록, 환경 프로퍼티     |
| **exception/** | 전역 예외 처리, 커스텀 예외 정의             |
| **util/**      | JSON 변환, 파일 입출력, 문자열 처리 등 유틸 함수 |
