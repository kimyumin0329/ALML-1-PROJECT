// src/main/java/com/watchserviceagent/watchservice_agent/collector/FileCollectorService.java
package com.watchserviceagent.watchservice_agent.collector;

import com.watchserviceagent.watchservice_agent.collector.business.EntropyAnalyzer;
import com.watchserviceagent.watchservice_agent.collector.dto.FileAnalysisResult;
import com.watchserviceagent.watchservice_agent.collector.snapshot.FileSnapshotStore;
import com.watchserviceagent.watchservice_agent.watcher.dto.WatcherEventRecord;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Instant;
import java.util.Locale;

@Service
@RequiredArgsConstructor
@Slf4j
public class FileCollectorService {

    private final EntropyAnalyzer entropyAnalyzer;
    private final FileSnapshotStore snapshotStore;

    public FileAnalysisResult analyze(WatcherEventRecord event) {
        String ownerKey = event.getOwnerKey();
        String eventType = event.getEventType();
        String pathStr = event.getPath();
        Instant eventTime = Instant.ofEpochMilli(event.getEventTimeMs());

        if (pathStr == null) {
            log.warn("WatcherEventRecord.path is null. event={}", event);
            return buildResultWithNoFile(ownerKey, eventType, eventTime, null);
        }

        Path path = Paths.get(pathStr);

        // ✅ before 기준: last가 있으면 last, 없으면 baseline
        FileSnapshotStore.Snapshot prev = snapshotStore.getLast(pathStr);
        if (prev == null) prev = snapshotStore.getBaseline(pathStr);

        boolean existsBefore = prev != null && prev.isExists();
        Long sizeBefore = prev != null ? prev.getSize() : null;
        Double entropyBefore = prev != null ? prev.getEntropy() : null;
        String extBefore = prev != null ? prev.getExt() : null;

        boolean existsNow = Files.exists(path) && Files.isRegularFile(path);

        Long sizeAfter = null;
        Double entropyAfter = null;
        String extAfter = null;
        Long lastModifiedAfter = null;

        if (existsNow && !"DELETE".equalsIgnoreCase(eventType)) {
            try {
                sizeAfter = Files.size(path);
            } catch (IOException e) {
                log.warn("Failed to get file size: {}", path, e);
            }

            try {
                entropyAfter = entropyAnalyzer.computeSampleEntropy(path, 4096);
            } catch (IOException e) {
                log.warn("Failed to compute entropy: {}", path, e);
            }

            extAfter = extractExtension(pathStr);

            try {
                lastModifiedAfter = Files.getLastModifiedTime(path).toMillis();
            } catch (IOException e) {
                log.warn("Failed to get lastModifiedTime: {}", path, e);
            }
        }

        FileAnalysisResult result = FileAnalysisResult.builder()
                .ownerKey(ownerKey)
                .eventType(eventType)
                .path(pathStr)
                .eventTime(eventTime)

                .existsBefore(existsBefore)
                .existsAfter(existsNow)

                .sizeBefore(sizeBefore)
                .sizeAfter(sizeAfter)
                .entropyBefore(entropyBefore)
                .entropyAfter(entropyAfter)
                .extBefore(extBefore)
                .extAfter(extAfter)

                .build();

        // ✅ snapshot 갱신 규칙
        FileSnapshotStore.Snapshot current = FileSnapshotStore.Snapshot.builder()
                .exists(existsNow)
                .size(sizeAfter)
                .entropy(entropyAfter)
                .ext(extAfter)
                .lastModifiedTime(lastModifiedAfter)
                .hash(null)
                .build();

        // baseline 없으면(즉시검사 이전 이벤트 등) 보정
        if (existsNow) {
            snapshotStore.putBaselineIfAbsent(pathStr, current);
        }

        if (!existsNow || "DELETE".equalsIgnoreCase(eventType)) {
            snapshotStore.removeLast(pathStr);
        } else {
            snapshotStore.putLast(pathStr, current);
            if ("CREATE".equalsIgnoreCase(eventType)) {
                snapshotStore.ensureBaselineOnCreate(pathStr, current);
            }
        }

        return result;
    }

    private FileAnalysisResult buildResultWithNoFile(String ownerKey, String eventType, Instant eventTime, String pathStr) {
        return FileAnalysisResult.builder()
                .ownerKey(ownerKey)
                .eventType(eventType)
                .path(pathStr)
                .eventTime(eventTime)
                .existsBefore(false)
                .existsAfter(false)
                .build();
    }

    private String extractExtension(String pathStr) {
        if (pathStr == null) return null;
        int lastDot = pathStr.lastIndexOf('.');
        if (lastDot < 0 || lastDot == pathStr.length() - 1) return null;
        String ext = pathStr.substring(lastDot + 1).trim().toLowerCase(Locale.ROOT);
        return ext.isEmpty() ? null : ext;
    }
}
