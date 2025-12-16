// src/main/java/com/watchserviceagent/watchservice_agent/analytics/EventWindowAggregator.java
package com.watchserviceagent.watchservice_agent.analytics;

import com.watchserviceagent.watchservice_agent.ai.AiService;
import com.watchserviceagent.watchservice_agent.ai.domain.AiResult;
import com.watchserviceagent.watchservice_agent.ai.dto.AiPayload;
import com.watchserviceagent.watchservice_agent.collector.dto.FileAnalysisResult;
import com.watchserviceagent.watchservice_agent.storage.LogService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.*;

/**
 * 여러 FileAnalysisResult 를 3초 윈도우로 묶어서 AI 서버에 보낼 피처(AiPayload)를 집계.
 *
 * ✅ NOTE:
 * - CREATE/MODIFY/DELETE 외 이벤트(예: SCAN)는 윈도우 집계/AI 분석에 태우지 않고 로그만 저장한다.
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class EventWindowAggregator {

    private static final long WINDOW_MS = 3000L;
    private static final long READ_SESSION_TIMEOUT_MS = 5 * 60 * 1000L;
    private static final long RENAME_MAX_GAP_MS = 2000L;

    private final AiService aiService;
    private final LogService logService;

    private Long currentWindowStartMs = null;
    private final List<FileAnalysisResult> currentEvents = new ArrayList<>();
    private final Map<String, Long> lastReadTimeByFile = new HashMap<>();

    private static class WindowStats {
        int fileReadCount;
        int fileWriteCount;
        int fileDeleteCount;
        int fileRenameCount;
        int fileEncryptLikeCount;
        int changedFilesCount;
        double entropyDiffMean;
        double sizeDiffMean;
        int randomExtensionFlag;
    }

    public synchronized void onFileAnalysisResult(FileAnalysisResult result) {
        String type = (result.getEventType() == null)
                ? "UNKNOWN"
                : result.getEventType().trim().toUpperCase(Locale.ROOT);

        // ✅ SCAN 등 비표준 이벤트는 윈도우 집계에서 제외(오탐 방지)
        if (!"CREATE".equals(type) && !"MODIFY".equals(type) && !"DELETE".equals(type)) {
            log.debug("[EventWindowAggregator] Non-watcher eventType={} -> save log only. path={}", type, result.getPath());
            logService.saveAsync(result);
            return;
        }

        long eventTimeMs = result.getEventTime() != null
                ? result.getEventTime().toEpochMilli()
                : System.currentTimeMillis();

        if (currentWindowStartMs == null) {
            currentWindowStartMs = eventTimeMs;
        } else if (eventTimeMs - currentWindowStartMs >= WINDOW_MS) {
            flushWindow();
            currentWindowStartMs = eventTimeMs;
        }

        currentEvents.add(result);
    }

    public synchronized void flushIfNeeded() {
        if (!currentEvents.isEmpty()) {
            flushWindow();
            currentWindowStartMs = null;
        }
    }

    private void flushWindow() {
        if (currentEvents.isEmpty()) return;

        WindowStats stats = computeWindowStats(currentEvents, lastReadTimeByFile);

        AiPayload payload = AiPayload.builder()
                .fileReadCount(stats.fileReadCount)
                .fileWriteCount(stats.fileWriteCount)
                .fileDeleteCount(stats.fileDeleteCount)
                .fileRenameCount(stats.fileRenameCount)
                .fileEncryptLikeCount(stats.fileEncryptLikeCount)
                .changedFilesCount(stats.changedFilesCount)
                .randomExtensionFlag(stats.randomExtensionFlag)
                .entropyDiffMean(stats.entropyDiffMean)
                .fileSizeDiffMean(stats.sizeDiffMean)
                .build();

        Instant windowStart = Instant.ofEpochMilli(currentWindowStartMs);
        Instant windowEnd = currentEvents.get(currentEvents.size() - 1).getEventTime();
        if (windowEnd == null) windowEnd = windowStart;

        log.info(
                "[EventWindowAggregator] 3초 윈도우 집계 완료. windowStart={}, windowEnd={} | read={}, write={}, delete={}, rename={}, encryptLike={}, changedFiles={}, entropyDiffMean={}, sizeDiffMean={}, randomExt={}",
                windowStart,
                windowEnd,
                stats.fileReadCount,
                stats.fileWriteCount,
                stats.fileDeleteCount,
                stats.fileRenameCount,
                stats.fileEncryptLikeCount,
                stats.changedFilesCount,
                stats.entropyDiffMean,
                stats.sizeDiffMean,
                stats.randomExtensionFlag
        );

        AiResult aiResult = aiService.requestAnalysis(payload);
        log.info("[EventWindowAggregator] AI 분석 결과: {}", aiResult);

        for (FileAnalysisResult r : currentEvents) {
            FileAnalysisResult enriched = r.withAiResult(aiResult);
            logService.saveAsync(enriched);
        }

        currentEvents.clear();
    }

    private WindowStats computeWindowStats(List<FileAnalysisResult> events, Map<String, Long> sessionState) {
        WindowStats stats = new WindowStats();

        int fileReadCount = 0;
        int fileWriteCount = 0;
        int fileDeleteCount = 0;
        int fileEncryptLikeCount = 0;

        Set<String> changedFileSet = new HashSet<>();

        double entropyDiffSum = 0.0;
        double sizeDiffSum = 0.0;
        int entropyDiffCount = 0;
        int sizeDiffCount = 0;

        double encryptEntropyThreshold = 0.3;
        double eps = 1e-6;

        for (FileAnalysisResult r : events) {
            String eventType = (r.getEventType() == null) ? "" : r.getEventType().trim().toUpperCase(Locale.ROOT);
            String path = r.getPath();
            String ownerKey = r.getOwnerKey();
            if (path != null) changedFileSet.add(path);

            Long sizeBefore = r.getSizeBefore();
            Long sizeAfter = r.getSizeAfter();
            Double entropyBefore = r.getEntropyBefore();
            Double entropyAfter = r.getEntropyAfter();

            long sizeDiff = 0L;
            if (sizeBefore != null && sizeAfter != null) {
                sizeDiff = sizeAfter - sizeBefore;
                sizeDiffSum += sizeDiff;
                sizeDiffCount++;
            }

            double entropyDiff = 0.0;
            if (entropyBefore != null && entropyAfter != null) {
                entropyDiff = entropyAfter - entropyBefore;
                entropyDiffSum += entropyDiff;
                entropyDiffCount++;
            }

            if ("MODIFY".equals(eventType)) {
                long ts = (r.getEventTime() != null)
                        ? r.getEventTime().toEpochMilli()
                        : System.currentTimeMillis();

                String key = ownerKey + "|" + path;

                Long lastRead = sessionState.get(key);
                boolean isNewRead = (lastRead == null) || (ts - lastRead > READ_SESSION_TIMEOUT_MS);

                if (isNewRead) {
                    fileReadCount++;
                    sessionState.put(key, ts);
                }

                boolean contentChanged = (sizeDiff != 0L) || (Math.abs(entropyDiff) > eps);
                if (contentChanged) {
                    fileWriteCount++;
                }
            }

            if ("DELETE".equals(eventType)) {
                fileDeleteCount++;
            }

            boolean extChanged = (r.getExtBefore() != null && r.getExtAfter() != null
                    && !r.getExtBefore().equalsIgnoreCase(r.getExtAfter()));
            if (entropyDiff > encryptEntropyThreshold && (sizeDiff != 0L || extChanged)) {
                fileEncryptLikeCount++;
            }
        }

        int fileRenameCount = detectRenameLikeCount(events);

        fileDeleteCount = Math.max(0, fileDeleteCount - fileRenameCount);
        fileWriteCount += fileRenameCount;

        int changedFilesCount = changedFileSet.size() - fileRenameCount;
        if (changedFilesCount < 0) changedFilesCount = 0;

        stats.fileReadCount = fileReadCount;
        stats.fileWriteCount = fileWriteCount;
        stats.fileDeleteCount = fileDeleteCount;
        stats.fileRenameCount = fileRenameCount;
        stats.fileEncryptLikeCount = fileEncryptLikeCount;
        stats.changedFilesCount = changedFilesCount;
        stats.entropyDiffMean = entropyDiffCount > 0 ? entropyDiffSum / entropyDiffCount : 0.0;
        stats.sizeDiffMean = sizeDiffCount > 0 ? sizeDiffSum / sizeDiffCount : 0.0;
        stats.randomExtensionFlag = detectRandomExtensionFlag(events);

        return stats;
    }

    private int detectRandomExtensionFlag(List<FileAnalysisResult> events) {
        String[] whiteList = {"txt", "log", "doc", "docx", "xls", "xlsx", "pdf",
                "png", "jpg", "jpeg", "gif", "zip", "rar", "7z"};

        for (FileAnalysisResult r : events) {
            if (isSuspiciousExt(r.getExtBefore(), whiteList) || isSuspiciousExt(r.getExtAfter(), whiteList)) {
                return 1;
            }
        }
        return 0;
    }

    private boolean isSuspiciousExt(String ext, String[] whiteList) {
        if (ext == null || ext.isBlank()) return false;
        String lower = ext.toLowerCase(Locale.ROOT);

        for (String w : whiteList) {
            if (w.equals(lower)) return false;
        }

        if (lower.length() < 4) return false;

        for (char c : lower.toCharArray()) {
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) return false;
        }
        return true;
    }

    private int detectRenameLikeCount(List<FileAnalysisResult> events) {
        List<FileAnalysisResult> deletes = new ArrayList<>();
        List<FileAnalysisResult> creates = new ArrayList<>();

        for (FileAnalysisResult r : events) {
            String t = (r.getEventType() == null) ? "" : r.getEventType().trim().toUpperCase(Locale.ROOT);
            if ("DELETE".equals(t)) deletes.add(r);
            else if ("CREATE".equals(t)) creates.add(r);
        }

        if (deletes.isEmpty() || creates.isEmpty()) return 0;

        int renameCount = 0;
        boolean[] usedCreate = new boolean[creates.size()];

        for (FileAnalysisResult del : deletes) {
            String ownerKey = del.getOwnerKey();
            Long delSize = del.getSizeBefore();
            String delExt = del.getExtBefore();
            if (delSize == null) continue;

            String delParent = getParentDir(del.getPath());
            long delMs = del.getEventTime() != null ? del.getEventTime().toEpochMilli() : 0L;

            for (int i = 0; i < creates.size(); i++) {
                if (usedCreate[i]) continue;

                FileAnalysisResult crt = creates.get(i);

                if (!Objects.equals(ownerKey, crt.getOwnerKey())) continue;
                if (!Objects.equals(delParent, getParentDir(crt.getPath()))) continue;

                Long crtSize = crt.getSizeAfter();
                if (crtSize == null) continue;
                if (!delSize.equals(crtSize)) continue;

                String crtExt = crt.getExtAfter();
                if (delExt != null && crtExt != null && !delExt.equalsIgnoreCase(crtExt)) continue;

                long crtMs = crt.getEventTime() != null ? crt.getEventTime().toEpochMilli() : 0L;
                long gap = Math.abs(crtMs - delMs);
                if (gap > RENAME_MAX_GAP_MS) continue;

                renameCount++;
                usedCreate[i] = true;
                break;
            }
        }

        return renameCount;
    }

    private String getParentDir(String path) {
        if (path == null) return null;
        int slash = path.lastIndexOf('/');
        int backslash = path.lastIndexOf('\\');
        int idx = Math.max(slash, backslash);
        if (idx < 0) return "";
        return path.substring(0, idx);
    }
}
