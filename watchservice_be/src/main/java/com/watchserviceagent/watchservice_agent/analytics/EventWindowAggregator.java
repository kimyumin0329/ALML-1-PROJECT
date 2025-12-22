package com.watchserviceagent.watchservice_agent.analytics;

import com.watchserviceagent.watchservice_agent.ai.AiService;
import com.watchserviceagent.watchservice_agent.ai.domain.AiResult;
import com.watchserviceagent.watchservice_agent.ai.dto.AiPayload;
import com.watchserviceagent.watchservice_agent.alerts.NotificationService;
import com.watchserviceagent.watchservice_agent.alerts.domain.Notification;
import com.watchserviceagent.watchservice_agent.collector.dto.FileAnalysisResult;
import com.watchserviceagent.watchservice_agent.common.util.SessionIdManager;
import com.watchserviceagent.watchservice_agent.storage.LogService;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;

import java.time.Instant;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 클래스 이름 : EventWindowAggregator
 * 기능 : 여러 FileAnalysisResult를 시간 윈도우(기본 3초)로 묶어서 AI 서버에 보낼 피처(AiPayload)를 집계한다.
 *        CREATE/MODIFY/DELETE 이벤트만 윈도우 집계에 포함하고, SCAN 등은 로그만 저장한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Component
@RequiredArgsConstructor
@Slf4j
public class EventWindowAggregator {

    // =========================
    // Config (application.yml)
    // =========================
    @Value("${watchservice.analytics.window-ms:3000}")
    private long windowMs;

    @Value("${watchservice.analytics.touch-session-timeout-ms:300000}") // 5분
    private long touchSessionTimeoutMs;

    @Value("${watchservice.analytics.rename-max-gap-ms:2000}")
    private long renameMaxGapMs;

    @Value("${watchservice.analytics.encrypt.entropy-diff-threshold:0.30}")
    private double encryptEntropyDiffThreshold;

    @Value("${watchservice.analytics.encrypt.min-size-bytes:4096}") // 4KB
    private long encryptMinSizeBytes;

    @Value("${watchservice.analytics.encrypt.eps:1.0E-6}")
    private double eps;

    @Value("${watchservice.analytics.random-ext.min-count:2}") // 윈도우 내 2개 이상이면 flag=1
    private int randomExtMinCount;

    @Value("${watchservice.analytics.random-ext.min-length:4}")
    private int randomExtMinLength;

    // csv: txt,log,doc,docx,...
    @Value("${watchservice.analytics.random-ext.whitelist:txt,log,doc,docx,xls,xlsx,pdf,png,jpg,jpeg,gif,zip,rar,7z}")
    private String randomExtWhitelistCsv;

    private Set<String> randomExtWhitelist = new HashSet<>();

    /**
     * 함수 이름 : init
     * 기능 : 애플리케이션 시작 시 확장자 화이트리스트를 초기화한다.
     * 매개변수 : 없음
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostConstruct
    void init() {
        randomExtWhitelist.clear();
        for (String s : String.valueOf(randomExtWhitelistCsv).split(",")) {
            String v = s.trim().toLowerCase(Locale.ROOT);
            if (!v.isBlank()) randomExtWhitelist.add(v);
        }
        log.info("[EventWindowAggregator] init config: windowMs={} touchTimeoutMs={} renameMaxGapMs={} encryptEntropyDiffThreshold={} encryptMinSizeBytes={} randomExtMinCount={} randomExtMinLength={} whitelistSize={}",
                windowMs, touchSessionTimeoutMs, renameMaxGapMs,
                encryptEntropyDiffThreshold, encryptMinSizeBytes,
                randomExtMinCount, randomExtMinLength, randomExtWhitelist.size());
    }

    // =========================
    // Deps
    // =========================
    private final AiService aiService;
    private final LogService logService;
    private final NotificationService notificationService;
    private final SessionIdManager sessionIdManager;

    // =========================
    // State
    // =========================
    private Long currentWindowStartMs = null;
    private final List<FileAnalysisResult> currentEvents = new ArrayList<>();

    // "touch"(접근) 세션 관리: ownerKey|path -> lastTouchMs
    private final Map<String, Long> lastTouchTimeByFile = new HashMap<>();

    private static class WindowStats {
        int fileTouchCount;          // "읽기"가 아니라 "최근에 접근/변경으로 감지된 파일 수(세션 기준)"
        int fileWriteCount;          // 내용 변경(크기/엔트로피 변화) 기반
        int fileDeleteCount;
        int fileRenameCount;
        int fileEncryptLikeCount;
        int changedFilesCount;
        double entropyDiffMean;
        double sizeDiffMean;

        int randomExtensionCount;    // 윈도우 내 suspicious ext (unique file 기준)
        int randomExtensionFlag;     // randomExtMinCount 이상이면 1
    }

    /**
     * 함수 이름 : onFileAnalysisResult
     * 기능 : 파일 분석 결과를 받아서 현재 윈도우에 추가한다. 윈도우 시간이 지나면 자동으로 flush하여 AI 분석을 수행한다.
     * 매개변수 : result - 파일 분석 결과
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public synchronized void onFileAnalysisResult(FileAnalysisResult result) {
        String type = normalizeType(result.getEventType());

        // ✅ SCAN 등 비표준 이벤트는 윈도우 집계에서 제외(오탐 방지)
        if (!"CREATE".equals(type) && !"MODIFY".equals(type) && !"DELETE".equals(type)) {
            log.debug("[EventWindowAggregator] Non-watcher eventType={} -> save log only. path={}", type, result.getPath());
            logService.saveAsync(result);
            return;
        }

        long eventTimeMs = (result.getEventTime() != null)
                ? result.getEventTime().toEpochMilli()
                : System.currentTimeMillis();

        if (currentWindowStartMs == null) {
            currentWindowStartMs = eventTimeMs;
        } else if (eventTimeMs - currentWindowStartMs >= windowMs) {
            flushWindow();
            currentWindowStartMs = eventTimeMs;
        }

        currentEvents.add(result);
    }

    /**
     * 함수 이름 : flushIfNeeded
     * 기능 : 현재 윈도우에 남아있는 이벤트가 있으면 강제로 flush하여 AI 분석을 수행한다.
     * 매개변수 : 없음
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public synchronized void flushIfNeeded() {
        if (!currentEvents.isEmpty()) {
            flushWindow();
            currentWindowStartMs = null;
        }
    }

    /**
     * 함수 이름 : flushWindow
     * 기능 : 현재 윈도우의 이벤트들을 집계하여 AI 서버에 전송하고, 결과를 로그에 저장한다.
     * 매개변수 : 없음
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private void flushWindow() {
        if (currentEvents.isEmpty()) return;

        WindowStats stats = computeWindowStats(currentEvents, lastTouchTimeByFile);

        // ✅ 9개 피처만 포함하는 AiPayload 빌드
        AiPayload payload = AiPayload.builder()
                .fileReadCount(stats.fileTouchCount)
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
                "[EventWindowAggregator] windowStart={} windowEnd={} | touch={}, write={}, delete={}, rename={}, encryptLike={}, changedFiles={}, entropyDiffMean={}, sizeDiffMean={}, randomExtCount={}, randomExtFlag={}",
                windowStart, windowEnd,
                stats.fileTouchCount,
                stats.fileWriteCount,
                stats.fileDeleteCount,
                stats.fileRenameCount,
                stats.fileEncryptLikeCount,
                stats.changedFilesCount,
                stats.entropyDiffMean,
                stats.sizeDiffMean,
                stats.randomExtensionCount,
                stats.randomExtensionFlag
        );

        AiResult aiResult = aiService.requestAnalysis(payload);
        
        // 랜섬웨어 감지 시 경고 로그
        if (aiResult.getIsRansomware() != null && aiResult.getIsRansomware()) {
            log.warn(
                    "[EventWindowAggregator] ⚠️ 랜섬웨어 감지! topFamily={}, score={}, label={}, detail={}",
                    aiResult.getTopFamily(), aiResult.getScore(), aiResult.getLabel(), aiResult.getDetail()
            );
        } else {
            log.info(
                    "[EventWindowAggregator] AI 분석 완료. topFamily={}, isRansomware={}, score={}, label={}",
                    aiResult.getTopFamily(), aiResult.getIsRansomware(), aiResult.getScore(), aiResult.getLabel()
            );
        }

        // ✅ 같은 윈도우 이벤트들에 AI 결과를 부착해서 저장
        for (FileAnalysisResult r : currentEvents) {
            FileAnalysisResult enriched = r.withAiResult(aiResult);
            logService.saveAsync(enriched);
        }

        // ✅ 윈도우 단위 알림 저장: affectedPaths 수집
        List<String> affectedPaths = currentEvents.stream()
                .map(FileAnalysisResult::getPath)
                .filter(Objects::nonNull)
                .distinct()
                .collect(Collectors.toList());

        String ownerKey = currentEvents.isEmpty() ? sessionIdManager.getSessionId() : currentEvents.get(0).getOwnerKey();
        if (ownerKey == null) ownerKey = sessionIdManager.getSessionId();

        Instant createdAt = Instant.now();

        Notification notification = Notification.builder()
                .ownerKey(ownerKey)
                .windowStart(windowStart)
                .windowEnd(windowEnd)
                .createdAt(createdAt)
                .aiLabel(aiResult.getLabel())
                .aiScore(aiResult.getScore())
                .topFamily(aiResult.getTopFamily())
                .aiDetail(aiResult.getDetail())
                .affectedFilesCount(affectedPaths.size())
                .affectedPaths(affectedPaths)
                .build();

        try {
            notificationService.saveNotification(notification);
            log.debug("[EventWindowAggregator] 알림 저장 완료: affectedFilesCount={}", affectedPaths.size());
        } catch (Exception e) {
            log.error("[EventWindowAggregator] 알림 저장 실패", e);
        }

        currentEvents.clear();
    }

    private WindowStats computeWindowStats(List<FileAnalysisResult> events, Map<String, Long> sessionState) {
        WindowStats stats = new WindowStats();

        int touchCount = 0;
        int writeCount = 0;
        int deleteCount = 0;
        int encryptLikeCount = 0;

        Set<String> changedFileSet = new HashSet<>();
        Set<String> suspiciousExtFileSet = new HashSet<>();

        double entropyDiffSum = 0.0;
        double sizeDiffSum = 0.0;
        int entropyDiffCount = 0;
        int sizeDiffCount = 0;

        for (FileAnalysisResult r : events) {
            String eventType = normalizeType(r.getEventType());
            String path = r.getPath();
            String ownerKey = r.getOwnerKey();

            if (path != null) changedFileSet.add(path);

            Long sizeBefore = r.getSizeBefore();
            Long sizeAfter = r.getSizeAfter();
            Double entropyBefore = r.getEntropyBefore();
            Double entropyAfter = r.getEntropyAfter();

            long sizeDiff = 0L;
            boolean hasSizePair = (sizeBefore != null && sizeAfter != null);
            if (hasSizePair) {
                sizeDiff = sizeAfter - sizeBefore;
                sizeDiffSum += sizeDiff;
                sizeDiffCount++;
            }

            double entropyDiff = 0.0;
            boolean hasEntropyPair = (entropyBefore != null && entropyAfter != null);
            if (hasEntropyPair) {
                entropyDiff = entropyAfter - entropyBefore;
                entropyDiffSum += entropyDiff;
                entropyDiffCount++;
            }

            // ---------------------------------------
            // MODIFY: touch/write 추정
            // ---------------------------------------
            if ("MODIFY".equals(eventType)) {
                long ts = (r.getEventTime() != null)
                        ? r.getEventTime().toEpochMilli()
                        : System.currentTimeMillis();

                String key = safe(ownerKey) + "|" + safe(path);

                Long lastTouch = sessionState.get(key);
                boolean isNewTouch = (lastTouch == null) || (ts - lastTouch > touchSessionTimeoutMs);
                if (isNewTouch) {
                    touchCount++;
                    sessionState.put(key, ts);
                }

                // "내용 변경" 기준 (너무 민감하지 않도록 eps 사용)
                boolean contentChanged = (hasSizePair && sizeDiff != 0L)
                        || (hasEntropyPair && Math.abs(entropyDiff) > eps);

                if (contentChanged) {
                    writeCount++;
                }
            }

            if ("DELETE".equals(eventType)) {
                deleteCount++;
            }

            // ---------------------------------------
            // random extension (윈도우 내 "여러개"면 플래그)
            // ---------------------------------------
            if (path != null) {
                if (isSuspiciousExt(r.getExtBefore()) || isSuspiciousExt(r.getExtAfter())) {
                    suspiciousExtFileSet.add(path);
                }
            }

            // ---------------------------------------
            // encrypt-like 판정 강화
            //  - 엔트로피 증가 + (크기변화 or 확장자변화)
            //  - 최소 크기 조건(작은 파일 엔트로피 튐 방지)
            // ---------------------------------------
            boolean extChanged = (r.getExtBefore() != null && r.getExtAfter() != null
                    && !r.getExtBefore().equalsIgnoreCase(r.getExtAfter()));

            boolean sizeChanged = (hasSizePair && sizeDiff != 0L);

            long approxSize = (sizeAfter != null) ? sizeAfter : (sizeBefore != null ? sizeBefore : 0L);
            boolean bigEnough = approxSize >= encryptMinSizeBytes;

            boolean entropyUp = (hasEntropyPair && entropyDiff >= encryptEntropyDiffThreshold);

            if (bigEnough && entropyUp && (sizeChanged || extChanged)) {
                encryptLikeCount++;
            }
        }

        // rename-like는 DELETE+CREATE를 점수 매칭으로 잡음 (미탐 줄이기)
        int renameCount = detectRenameLikeCountByScore(events);

        // rename은 DELETE로도 잡히고 CREATE로도 잡히는 경우가 많아서
        // delete에서 빼고, write로 포함(변경 이벤트로 취급)
        deleteCount = Math.max(0, deleteCount - renameCount);
        writeCount += renameCount;

        int changedFilesCount = changedFileSet.size() - renameCount;
        if (changedFilesCount < 0) changedFilesCount = 0;

        int suspiciousExtCount = suspiciousExtFileSet.size();
        int randomExtFlag = (suspiciousExtCount >= Math.max(1, randomExtMinCount)) ? 1 : 0;

        stats.fileTouchCount = touchCount;
        stats.fileWriteCount = writeCount;
        stats.fileDeleteCount = deleteCount;
        stats.fileRenameCount = renameCount;
        stats.fileEncryptLikeCount = encryptLikeCount;
        stats.changedFilesCount = changedFilesCount;
        stats.entropyDiffMean = (entropyDiffCount > 0) ? (entropyDiffSum / entropyDiffCount) : 0.0;
        stats.sizeDiffMean = (sizeDiffCount > 0) ? (sizeDiffSum / sizeDiffCount) : 0.0;

        stats.randomExtensionCount = suspiciousExtCount;
        stats.randomExtensionFlag = randomExtFlag;

        return stats;
    }

    /**
     * rename-like: DELETE와 CREATE를 "점수"로 매칭해서 미탐을 줄인다.
     *
     * 필수:
     * - ownerKey 동일
     * - parent dir 동일
     * - 시간 gap <= renameMaxGapMs
     *
     * 점수:
     * - size 동일이면 +2
     * - ext 동일이면 +1
     * - size가 둘 다 있고 "근사 동일(±1%)"이면 +1
     * => score >= 3 이면 rename으로 간주
     */
    private int detectRenameLikeCountByScore(List<FileAnalysisResult> events) {
        List<FileAnalysisResult> deletes = new ArrayList<>();
        List<FileAnalysisResult> creates = new ArrayList<>();

        for (FileAnalysisResult r : events) {
            String t = normalizeType(r.getEventType());
            if ("DELETE".equals(t)) deletes.add(r);
            else if ("CREATE".equals(t)) creates.add(r);
        }

        if (deletes.isEmpty() || creates.isEmpty()) return 0;

        int renameCount = 0;
        boolean[] usedCreate = new boolean[creates.size()];

        for (FileAnalysisResult del : deletes) {
            String ownerKey = del.getOwnerKey();
            String delPath = del.getPath();
            if (delPath == null) continue;

            Long delSize = del.getSizeBefore();
            String delExt = del.getExtBefore();
            String delParent = getParentDir(delPath);
            long delMs = (del.getEventTime() != null) ? del.getEventTime().toEpochMilli() : 0L;

            int bestIdx = -1;
            int bestScore = -1;

            for (int i = 0; i < creates.size(); i++) {
                if (usedCreate[i]) continue;

                FileAnalysisResult crt = creates.get(i);
                if (!Objects.equals(ownerKey, crt.getOwnerKey())) continue;

                String crtPath = crt.getPath();
                if (crtPath == null) continue;

                if (!Objects.equals(delParent, getParentDir(crtPath))) continue;

                long crtMs = (crt.getEventTime() != null) ? crt.getEventTime().toEpochMilli() : 0L;
                long gap = Math.abs(crtMs - delMs);
                if (gap > renameMaxGapMs) continue;

                int score = 0;

                Long crtSize = crt.getSizeAfter();
                if (delSize != null && crtSize != null) {
                    if (delSize.equals(crtSize)) score += 2;
                    else {
                        // 근사 동일(±1%)
                        long denom = Math.max(1L, delSize);
                        double ratio = Math.abs(crtSize - delSize) / (double) denom;
                        if (ratio <= 0.01) score += 1;
                    }
                }

                String crtExt = crt.getExtAfter();
                if (delExt != null && crtExt != null && delExt.equalsIgnoreCase(crtExt)) score += 1;

                // score 3 이상이면 rename 후보
                if (score >= 3 && score > bestScore) {
                    bestScore = score;
                    bestIdx = i;
                }
            }

            if (bestIdx >= 0) {
                usedCreate[bestIdx] = true;
                renameCount++;
            }
        }

        return renameCount;
    }

    private boolean isSuspiciousExt(String ext) {
        if (ext == null || ext.isBlank()) return false;
        String lower = ext.trim().toLowerCase(Locale.ROOT);

        if (randomExtWhitelist.contains(lower)) return false;
        if (lower.length() < Math.max(1, randomExtMinLength)) return false;

        // "영숫자만"으로 이루어진 확장자만 대상으로 (랜섬웨어 랜덤 확장자 패턴)
        for (char c : lower.toCharArray()) {
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9'))) return false;
        }
        return true;
    }

    private String getParentDir(String path) {
        if (path == null) return null;
        int slash = path.lastIndexOf('/');
        int backslash = path.lastIndexOf('\\');
        int idx = Math.max(slash, backslash);
        if (idx < 0) return "";
        return path.substring(0, idx);
    }

    private String normalizeType(String eventType) {
        if (eventType == null) return "UNKNOWN";
        return eventType.trim().toUpperCase(Locale.ROOT);
    }

    private String safe(String s) {
        return (s == null) ? "" : s;
    }
}
