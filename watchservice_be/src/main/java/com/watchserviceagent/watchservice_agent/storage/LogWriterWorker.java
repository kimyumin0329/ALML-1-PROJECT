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
 * Collector/Analytics → Storage 구간의 비동기 큐 역할을 하는 워커.
 *
 * - FileAnalysisResult 를 큐에 넣으면,
 * - 별도 워커 스레드가 이를 하나씩 꺼내어 SQLite 에 Log 로 INSERT 한다.
 *
 * 업데이트:
 *  - FileAnalysisResult 에 포함된 aiLabel / aiScore / aiDetail 도 함께 저장한다.
 */
@Slf4j
@Component
@RequiredArgsConstructor
public class LogWriterWorker {

    private final LogRepository logRepository;

    private final BlockingQueue<FileAnalysisResult> queue = new LinkedBlockingQueue<>();

    private Thread workerThread;
    private volatile boolean running = true;

    @PostConstruct
    public void start() {
        workerThread = new Thread(this::runWorker, "LogWriterWorker-Thread");
        workerThread.start();
        log.info("[LogWriterWorker] 워커 스레드 시작");
    }

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
     * FileAnalysisResult 를 큐에 넣는 진입점.
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
     * FileAnalysisResult → Log 엔티티 매핑.
     * (AI 결과 포함)
     */
    private Log mapToLog(FileAnalysisResult r) {
        Instant collectedAt = r.getEventTime();
        if (collectedAt == null) {
            collectedAt = Instant.now();
        }

        Long sizeAfter = r.getSizeAfter();
        Double entropyAfter = r.getEntropyAfter();
        boolean existsAfter = r.isExistsAfter();

        long sizeForLog = (sizeAfter != null) ? sizeAfter : -1L;
        long lastModifiedForLog = collectedAt.toEpochMilli();

        return Log.builder()
                .id(null)  // AUTOINCREMENT
                .ownerKey(r.getOwnerKey())
                .eventType(r.getEventType())
                .path(r.getPath())
                .exists(existsAfter)
                .size(sizeForLog)
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
