package com.watchserviceagent.watchservice_agent.storage;

import com.watchserviceagent.watchservice_agent.collector.dto.FileAnalysisResult;
import com.watchserviceagent.watchservice_agent.storage.domain.Log;
import jakarta.annotation.PostConstruct;
import jakarta.annotation.PreDestroy;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;

/**
 * 클래스 이름 : LogWriterWorker
 * 기능 : Collector/Analytics에서 생성된 FileAnalysisResult를 비동기로 큐에 넣어 SQLite에 저장하는 워커 스레드를 관리한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class LogWriterWorker {

    private final LogRepository logRepository;

    private final BlockingQueue<FileAnalysisResult> queue = new LinkedBlockingQueue<>();

    private Thread workerThread;
    private volatile boolean running = true;

    /**
     * 함수 이름 : start
     * 기능 : 워커 스레드를 시작한다. 애플리케이션 시작 시 자동 호출된다.
     * 매개변수 : 없음
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostConstruct
    public void start() {
        workerThread = new Thread(this::runWorker, "LogWriterWorker-Thread");
        workerThread.start();
        log.info("[LogWriterWorker] 워커 스레드 시작");
    }

    /**
     * 함수 이름 : stop
     * 기능 : 워커 스레드를 중지하고 리소스를 정리한다. 애플리케이션 종료 시 자동 호출된다.
     * 매개변수 : 없음
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PreDestroy
    public void stop() {
        running = false;
        if (workerThread != null) {
            workerThread.interrupt();
            try {
                workerThread.join(3000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.warn("[LogWriterWorker] stop() join 중 인터럽트", e);
            }
        }
        log.info("[LogWriterWorker] 워커 스레드 종료");
    }

    /**
     * 함수 이름 : enqueue
     * 기능 : FileAnalysisResult를 비동기 저장 큐에 추가한다.
     * 매개변수 : result - 저장할 파일 분석 결과
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public void enqueue(FileAnalysisResult result) {
        if (result == null) return;
        try {
            queue.put(result);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            log.warn("[LogWriterWorker] enqueue 중 인터럽트 발생", e);
        }
    }

    /**
     * 함수 이름 : runWorker
     * 기능 : 워커 스레드의 메인 루프. 큐에서 FileAnalysisResult를 꺼내어 DB에 저장한다.
     * 매개변수 : 없음
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private void runWorker() {
        while (running && !Thread.currentThread().isInterrupted()) {
            try {
                FileAnalysisResult result = queue.take();
                Log logEntity = mapToLog(result);
                logRepository.insertLog(logEntity);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.info("[LogWriterWorker] 워커 스레드 인터럽트, 종료 준비");
            } catch (Exception e) {
                log.error("[LogWriterWorker] 로그 저장 중 예외 발생", e);
            }
        }
        log.info("[LogWriterWorker] 워커 루프 종료");
    }

    /**
     * 함수 이름 : mapToLog
     * 기능 : FileAnalysisResult를 Log 엔티티로 변환한다. AI 분석 결과도 포함한다.
     * 매개변수 : r - 파일 분석 결과
     * 반환값 : Log - 로그 엔티티
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private Log mapToLog(FileAnalysisResult r) {
        Instant collectedAt = r.getEventTime();
        if (collectedAt == null) {
            collectedAt = Instant.now();
        }

        Long sizeAfter = r.getSizeAfter();
        Double entropyAfter = r.getEntropyAfter();
        boolean existsAfter = r.isExistsAfter();
        Long sizeBefore = r.getSizeBefore();
        Double entropyBefore = r.getEntropyBefore();
        Long sizeDiff = r.getSizeDiff();
        Double entropyDiff = r.getEntropyDiff();

        long sizeForLog = (sizeAfter != null) ? sizeAfter : -1L;
        long lastModifiedForLog = collectedAt.toEpochMilli();

        return Log.builder()
                .id(null)  // AUTOINCREMENT
                .ownerKey(r.getOwnerKey())
                .eventType(r.getEventType())
                .path(r.getPath())
                .exists(existsAfter)
                .size(sizeForLog)
                .sizeBefore(sizeBefore)
                .sizeAfter(sizeAfter)
                .entropyBefore(entropyBefore)
                .entropyAfter(entropyAfter)
                .extBefore(r.getExtBefore())
                .extAfter(r.getExtAfter())
                .existsBefore(r.isExistsBefore())
                .sizeDiff(sizeDiff)
                .entropyDiff(entropyDiff)
                .lastModifiedTime(lastModifiedForLog)
                .hash(null)              // hash 는 아직 미사용
                .entropy(entropyAfter)

                // === 여기서 AI 결과도 같이 저장 ===
                .aiLabel(r.getAiLabel())
                .aiScore(r.getAiScore())
                .aiDetail(r.getAiDetail())

                .collectedAt(collectedAt)
                .build();
    }
}
