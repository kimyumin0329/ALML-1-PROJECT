package com.watchserviceagent.watchservice_agent.storage;

import com.watchserviceagent.watchservice_agent.collector.dto.FileAnalysisResult;
import com.watchserviceagent.watchservice_agent.common.util.SessionIdManager;
import com.watchserviceagent.watchservice_agent.storage.domain.Log;
import com.watchserviceagent.watchservice_agent.storage.dto.LogExportRequest;
import com.watchserviceagent.watchservice_agent.storage.dto.LogPageResponse;
import com.watchserviceagent.watchservice_agent.storage.dto.LogResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.List;

/**
 * 클래스 이름 : LogService
 * 기능 : 로그 저장, 조회, 삭제, 내보내기 등의 비즈니스 로직을 처리한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class LogService {

    private final SessionIdManager sessionIdManager;
    private final LogWriterWorker logWriterWorker;
    private final LogRepository logRepository;

    private static final int MAX_PAGE_SIZE = 1000;
    private static final int EXPORT_LIMIT = 20000;
    private static final DateTimeFormatter DATE_TIME_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    /**
     * 함수 이름 : saveAsync
     * 기능 : 파일 분석 결과를 비동기로 로그 저장 큐에 추가한다.
     * 매개변수 : result - 파일 분석 결과
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public void saveAsync(FileAnalysisResult result) {
        if (result == null) return;
        logWriterWorker.enqueue(result);
    }

    /**
     * 함수 이름 : getRecentLogs
     * 기능 : 최근 로그를 지정된 개수만큼 조회한다.
     * 매개변수 : limit - 조회할 로그 개수
     * 반환값 : LogResponse 리스트
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public List<LogResponse> getRecentLogs(int limit) {
        String ownerKey = sessionIdManager.getSessionId();
        List<Log> logs = logRepository.findRecentLogsByOwner(ownerKey, limit);
        return logs.stream().map(LogResponse::from).toList();
    }

    /**
     * 함수 이름 : getLogs
     * 기능 : 페이지네이션, 필터링, 정렬을 지원하는 로그 목록을 조회한다.
     * 매개변수 : page - 페이지 번호, size - 페이지 크기, from - 시작 날짜, to - 종료 날짜, keyword - 검색 키워드, aiLabel - AI 라벨 필터, eventType - 이벤트 타입 필터, sort - 정렬 기준
     * 반환값 : LogPageResponse - 페이지네이션된 로그 목록
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public LogPageResponse getLogs(Integer page, Integer size, String from, String to,
                                   String keyword, String aiLabel, String eventType, String sort) {

        int p = (page == null || page < 1) ? 1 : page;
        int s = (size == null || size < 1) ? 50 : Math.min(size, MAX_PAGE_SIZE);

        SortInfo sortInfo = parseSort(sort);
        Long fromEpoch = parseFromToEpochStart(from);
        Long toEpoch = parseFromToEpochEnd(to);

        String ownerKey = sessionIdManager.getSessionId();

        long total = logRepository.countLogsByOwner(ownerKey, fromEpoch, toEpoch, keyword, aiLabel, eventType);
        int offset = (p - 1) * s;

        List<Log> rows = logRepository.findLogsByOwner(
                ownerKey,
                fromEpoch,
                toEpoch,
                keyword,
                aiLabel,
                eventType,
                sortInfo.field,
                sortInfo.dir,
                offset,
                s
        );

        List<LogResponse> items = rows.stream().map(LogResponse::from).toList();

        return LogPageResponse.builder()
                .items(items)
                .page(p)
                .size(s)
                .total(total)
                .build();
    }

    /**
     * 함수 이름 : getLogById
     * 기능 : ID로 단일 로그를 조회한다.
     * 매개변수 : id - 로그 ID
     * 반환값 : LogResponse - 로그 상세 정보
     * 예외 : IllegalArgumentException - 로그를 찾을 수 없을 때
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public LogResponse getLogById(long id) {
        String ownerKey = sessionIdManager.getSessionId();
        Log log = logRepository.findByIdAndOwner(ownerKey, id)
                .orElseThrow(() -> new IllegalArgumentException("log not found: id=" + id));
        return LogResponse.from(log);
    }

    /**
     * 함수 이름 : deleteOne
     * 기능 : 단일 로그를 삭제한다.
     * 매개변수 : id - 삭제할 로그 ID
     * 반환값 : 없음
     * 예외 : IllegalArgumentException - 로그를 찾을 수 없을 때
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public void deleteOne(long id) {
        String ownerKey = sessionIdManager.getSessionId();
        int deleted = logRepository.deleteByIdAndOwner(ownerKey, id);
        if (deleted <= 0) throw new IllegalArgumentException("log not found: id=" + id);
    }

    /**
     * 함수 이름 : deleteMany
     * 기능 : 여러 로그를 일괄 삭제한다.
     * 매개변수 : ids - 삭제할 로그 ID 리스트
     * 반환값 : 삭제된 로그 개수
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public int deleteMany(List<Long> ids) {
        if (ids == null || ids.isEmpty()) return 0;
        String ownerKey = sessionIdManager.getSessionId();
        return logRepository.deleteByIdsAndOwner(ownerKey, ids);
    }

    public ExportResult exportLogs(LogExportRequest req) {
        String ownerKey = sessionIdManager.getSessionId();

        List<Long> ids = req.getIds();
        String format = (req.getFormat() == null) ? "CSV" : req.getFormat().trim().toUpperCase();

        List<Log> logs;
        if (ids != null && !ids.isEmpty()) {
            logs = logRepository.findByIdsAndOwner(ownerKey, ids, EXPORT_LIMIT);
        } else {
            LogExportRequest.Filters f = req.getFilters();
            Long fromEpoch = parseFromToEpochStart(f == null ? null : f.getFrom());
            Long toEpoch = parseFromToEpochEnd(f == null ? null : f.getTo());
            String keyword = f == null ? null : f.getKeyword();
            String aiLabel = f == null ? null : f.getAiLabel();
            String eventType = f == null ? null : f.getEventType();
            SortInfo sortInfo = parseSort(f == null ? null : f.getSort());

            logs = logRepository.findLogsByOwner(
                    ownerKey,
                    fromEpoch,
                    toEpoch,
                    keyword,
                    aiLabel,
                    eventType,
                    sortInfo.field,
                    sortInfo.dir,
                    0,
                    EXPORT_LIMIT
            );
        }

        if ("JSON".equals(format)) {
            List<LogResponse> jsonItems = logs.stream().map(LogResponse::from).toList();
            return ExportResult.json(jsonItems);
        }

        String csv = toCsv(logs);
        return ExportResult.csv(csv);
    }

    private String toCsv(List<Log> logs) {
        StringBuilder sb = new StringBuilder();
        sb.append("id,collectedAt,eventType,path,exists,size,sizeBefore,sizeAfter,sizeDiff,entropy,entropyBefore,entropyAfter,entropyDiff,extBefore,extAfter,lastModifiedTime,hash,aiLabel,aiScore,aiDetail\n");

        for (Log log : logs) {
            LogResponse r = LogResponse.from(log);
            sb.append(r.getId()).append(",");
            sb.append(csvEsc(r.getCollectedAt())).append(",");
            sb.append(csvEsc(r.getEventType())).append(",");
            sb.append(csvEsc(r.getPath())).append(",");
            sb.append(r.isExists()).append(",");
            sb.append(r.getSize()).append(",");
            sb.append(r.getSizeBefore() == null ? "" : r.getSizeBefore()).append(",");
            sb.append(r.getSizeAfter() == null ? "" : r.getSizeAfter()).append(",");
            sb.append(r.getSizeDiff() == null ? "" : r.getSizeDiff()).append(",");
            sb.append(csvEsc(r.getLastModifiedTime())).append(",");
            sb.append(csvEsc(r.getHash())).append(",");
            sb.append(r.getEntropy() == null ? "" : r.getEntropy()).append(",");
            sb.append(r.getEntropyBefore() == null ? "" : r.getEntropyBefore()).append(",");
            sb.append(r.getEntropyAfter() == null ? "" : r.getEntropyAfter()).append(",");
            sb.append(r.getEntropyDiff() == null ? "" : r.getEntropyDiff()).append(",");
            sb.append(csvEsc(r.getExtBefore())).append(",");
            sb.append(csvEsc(r.getExtAfter())).append(",");
            sb.append(csvEsc(r.getAiLabel())).append(",");
            sb.append(r.getAiScore() == null ? "" : r.getAiScore()).append(",");
            sb.append(csvEsc(r.getAiDetail()));
            sb.append("\n");
        }

        return sb.toString();
    }

    private String csvEsc(String s) {
        if (s == null) return "";
        boolean needQuote = s.contains(",") || s.contains("\"") || s.contains("\n");
        String v = s.replace("\"", "\"\"");
        return needQuote ? ("\"" + v + "\"") : v;
    }

    private Long parseFromToEpochStart(String from) {
        if (from == null || from.isBlank()) return null;

        String v = from.trim();
        try {
            LocalDate d = LocalDate.parse(v);
            return d.atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (DateTimeParseException ignore) {}

        try {
            LocalDateTime dt = LocalDateTime.parse(v, DATE_TIME_FMT);
            return dt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (DateTimeParseException ignore) {}

        return null;
    }

    private Long parseFromToEpochEnd(String to) {
        if (to == null || to.isBlank()) return null;

        String v = to.trim();
        try {
            LocalDate d = LocalDate.parse(v);
            return d.plusDays(1).atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (DateTimeParseException ignore) {}

        try {
            LocalDateTime dt = LocalDateTime.parse(v, DATE_TIME_FMT);
            return dt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (DateTimeParseException ignore) {}

        return null;
    }

    private SortInfo parseSort(String sort) {
        String field = "collectedAt";
        String dir = "DESC";

        if (sort != null && !sort.isBlank()) {
            String[] parts = sort.split(",");
            if (parts.length >= 1 && !parts[0].isBlank()) field = parts[0].trim();
            if (parts.length >= 2 && !parts[1].isBlank()) dir = parts[1].trim().toUpperCase();
        }

        if (!dir.equals("ASC") && !dir.equals("DESC")) dir = "DESC";
        return new SortInfo(field, dir);
    }

    private record SortInfo(String field, String dir) {}

    @lombok.Getter
    @lombok.Builder
    public static class ExportResult {
        private final boolean json;
        private final List<LogResponse> jsonItems;
        private final String csvText;

        public static ExportResult json(List<LogResponse> items) {
            return ExportResult.builder().json(true).jsonItems(items).build();
        }

        public static ExportResult csv(String csvText) {
            return ExportResult.builder().json(false).csvText(csvText).build();
        }

        public boolean isJson() { return json; }
        public List<LogResponse> getJsonItems() { return jsonItems; }
        public String getCsvText() { return csvText; }
    }
}
