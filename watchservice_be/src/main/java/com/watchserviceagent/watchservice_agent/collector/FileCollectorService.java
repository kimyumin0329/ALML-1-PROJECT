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

/**
 * 클래스 이름 : FileCollectorService
 * 기능 : 파일 이벤트를 분석하여 이전 상태와 현재 상태를 비교하고, 파일 크기/엔트로피/확장자 변화를 계산한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class FileCollectorService {

    private final EntropyAnalyzer entropyAnalyzer;
    private final FileSnapshotStore snapshotStore;

    /**
     * 함수 이름 : analyze
     * 기능 : 파일 이벤트를 분석하여 이전 상태(before)와 현재 상태(after)를 비교한 결과를 반환한다.
     * 매개변수 : event - Watcher에서 발생한 파일 이벤트 레코드
     * 반환값 : FileAnalysisResult - 파일 분석 결과 (크기, 엔트로피, 확장자 변화 포함)
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
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

    /**
     * 함수 이름 : buildResultWithNoFile
     * 기능 : 파일이 존재하지 않는 경우의 기본 분석 결과를 생성한다.
     * 매개변수 : ownerKey - 세션 식별자, eventType - 이벤트 타입, eventTime - 이벤트 발생 시각, pathStr - 파일 경로
     * 반환값 : FileAnalysisResult - 기본값으로 채워진 분석 결과
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
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

    /**
     * 함수 이름 : extractExtension
     * 기능 : 파일 경로에서 확장자를 추출한다.
     * 매개변수 : pathStr - 파일 경로 문자열
     * 반환값 : 확장자 문자열 (소문자), 없으면 null
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private String extractExtension(String pathStr) {
        if (pathStr == null) return null;
        int lastDot = pathStr.lastIndexOf('.');
        if (lastDot < 0 || lastDot == pathStr.length() - 1) return null;
        String ext = pathStr.substring(lastDot + 1).trim().toLowerCase(Locale.ROOT);
        return ext.isEmpty() ? null : ext;
    }
}
