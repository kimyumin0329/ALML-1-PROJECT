package com.watchserviceagent.watchservice_agent.scan;

import com.watchserviceagent.watchservice_agent.collector.FileCollectorService;
import com.watchserviceagent.watchservice_agent.collector.dto.FileAnalysisResult;
import com.watchserviceagent.watchservice_agent.common.util.SessionIdManager;
import com.watchserviceagent.watchservice_agent.scan.domain.ScanJob;
import com.watchserviceagent.watchservice_agent.scan.dto.ScanProgressResponse;
import com.watchserviceagent.watchservice_agent.watcher.WatcherService;
import com.watchserviceagent.watchservice_agent.watcher.dto.WatcherEventRecord;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.*;

@Service
@RequiredArgsConstructor
@Slf4j
public class ScanService {

    private final SessionIdManager sessionIdManager;
    private final FileCollectorService fileCollectorService;
    private final WatcherService watcherService;

    private final Map<String, ScanJob> jobs = new ConcurrentHashMap<>();
    private final ExecutorService executor = Executors.newCachedThreadPool();

    public String startScan(List<String> paths, boolean autoStartWatcher) {
        List<String> roots = (paths == null) ? List.of() : paths.stream()
                .filter(Objects::nonNull)
                .map(String::trim)
                .filter(s -> !s.isBlank())
                .distinct()
                .toList();

        if (roots.isEmpty()) {
            throw new IllegalArgumentException("scan paths is empty");
        }

        String scanId = UUID.randomUUID().toString();
        ScanJob job = new ScanJob(scanId, roots);
        jobs.put(scanId, job);

        executor.submit(() -> runScan(job, autoStartWatcher));

        return scanId;
    }

    public void pause(String scanId) {
        ScanJob job = getJobOrThrow(scanId);
        job.pause();
    }

    public ScanProgressResponse getProgress(String scanId) {
        ScanJob job = getJobOrThrow(scanId);

        return ScanProgressResponse.builder()
                .status(job.getStatus().name())
                .percent(job.getPercent())
                .scanned(job.getScanned().get())
                .total(job.getTotal().get())
                .currentPath(job.getCurrentPath())
                .message(job.getMessage())
                .build();
    }

    private ScanJob getJobOrThrow(String scanId) {
        ScanJob job = jobs.get(scanId);
        if (job == null) throw new IllegalArgumentException("scan not found: " + scanId);
        return job;
    }

    private void runScan(ScanJob job, boolean autoStartWatcher) {
        String ownerKey = sessionIdManager.getSessionId();

        try {
            // 1) total 계산(진행률용)
            long total = countTotalFiles(job.getRoots());
            job.setTotal(total);
            log.info("[ScanService] scanId={} totalFiles={}", job.getScanId(), total);

            // 2) 실제 스캔 수행(기준값 등록 = snapshot baseline/last 채움)
            for (String rootStr : job.getRoots()) {
                if (job.isStopRequested()) break;

                Path root;
                try {
                    root = Paths.get(rootStr);
                } catch (Exception e) {
                    log.warn("[ScanService] invalid root path string: {}", rootStr);
                    continue;
                }

                if (!Files.exists(root) || !Files.isDirectory(root)) {
                    log.warn("[ScanService] skip invalid root: {}", rootStr);
                    continue;
                }

                try (var stream = Files.walk(root)) {
                    Iterator<Path> it = stream
                            .filter(Files::isRegularFile)
                            .iterator();

                    while (it.hasNext()) {
                        if (job.isStopRequested()) break;

                        Path path = it.next();
                        String p = path.toAbsolutePath().toString();
                        job.setCurrentPath(p);

                        // eventType=SCAN : Collector가 snapshot baseline/last 채우도록 함
                        WatcherEventRecord rec = WatcherEventRecord.builder()
                                .ownerKey(ownerKey)
                                .eventType("SCAN")
                                .path(p)
                                .eventTimeMs(System.currentTimeMillis())
                                .build();

                        // ✅ 중요한 점: SCAN은 “로그 저장/AI 집계” 하지 않는다 (폭발 방지)
                        // Collector 내부에서 snapshot만 채워짐
                        FileAnalysisResult ignored = fileCollectorService.analyze(rec);

                        job.incScanned();
                    }
                }
            }

            if (job.isStopRequested()) {
                log.info("[ScanService] scanId={} PAUSED scanned={}/{}", job.getScanId(), job.getScanned().get(), job.getTotal().get());
                return;
            }

            job.done();
            log.info("[ScanService] scanId={} DONE scanned={}/{}", job.getScanId(), job.getScanned().get(), job.getScanned().get());

            // 3) scan 완료 후 watcher 자동 시작 (유효 root만!)
            if (autoStartWatcher) {
                List<String> validRoots = job.getRoots().stream()
                        .filter(Objects::nonNull)
                        .map(String::trim)
                        .filter(s -> !s.isBlank())
                        .filter(s -> {
                            try {
                                Path p = Paths.get(s);
                                return Files.exists(p) && Files.isDirectory(p);
                            } catch (Exception e) {
                                return false;
                            }
                        })
                        .toList();

                if (!validRoots.isEmpty()) {
                    try {
                        watcherService.startWatchingMultiple(validRoots);
                        log.info("[ScanService] scanId={} watcher auto-start OK roots={}", job.getScanId(), validRoots);
                    } catch (Exception e) {
                        log.warn("[ScanService] watcher auto-start failed: {}", e.getMessage(), e);
                    }
                } else {
                    log.warn("[ScanService] scanId={} watcher auto-start skipped (no valid roots)", job.getScanId());
                }
            }

        } catch (Exception e) {
            log.error("[ScanService] scanId={} ERROR", job.getScanId(), e);
            job.error(e.getMessage());
        } finally {
            job.setCurrentPath(null);
        }
    }

    private long countTotalFiles(List<String> roots) {
        long total = 0;
        for (String rootStr : (roots == null ? List.<String>of() : roots)) {
            try {
                Path root = Paths.get(rootStr);
                if (!Files.exists(root) || !Files.isDirectory(root)) continue;

                try (var stream = Files.walk(root)) {
                    total += stream.filter(Files::isRegularFile).count();
                }
            } catch (IOException ignore) {
            } catch (Exception ignore) {
            }
        }
        return total;
    }
}
