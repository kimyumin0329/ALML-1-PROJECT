package com.watchserviceagent.watchservice_agent.storage;

import com.watchserviceagent.watchservice_agent.storage.dto.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.nio.charset.StandardCharsets;
import java.util.List;

@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
@RestController
@RequestMapping("/logs")
@RequiredArgsConstructor
@Slf4j
public class LogController {

    private final LogService logService;

    // ✅ 기존 유지: 최근 로그
    @GetMapping("/recent")
    public List<LogResponse> getRecentLogs(@RequestParam(name = "limit", defaultValue = "50") int limit) {
        if (limit <= 0) limit = 50;
        else if (limit > 1000) limit = 1000;

        List<LogResponse> logs = logService.getRecentLogs(limit);
        log.info("[LogController] GET /logs/recent limit={} -> {}", limit, logs.size());
        return logs;
    }

    // 🆕 로그 목록(감시 이벤트): 페이지/필터
    @GetMapping
    public LogPageResponse getLogs(
            @RequestParam(name = "page", required = false) Integer page,
            @RequestParam(name = "size", required = false) Integer size,
            @RequestParam(name = "from", required = false) String from,
            @RequestParam(name = "to", required = false) String to,
            @RequestParam(name = "keyword", required = false) String keyword,
            @RequestParam(name = "aiLabel", required = false) String aiLabel,
            @RequestParam(name = "eventType", required = false) String eventType,
            @RequestParam(name = "sort", required = false) String sort
    ) {
        return logService.getLogs(page, size, from, to, keyword, aiLabel, eventType, sort);
    }

    // 🆕 로그 상세(감시 이벤트): /logs/{id}
    @GetMapping("/{id}")
    public LogResponse getLog(@PathVariable("id") long id) {
        return logService.getLogById(id);
    }

    // 🆕 단건 삭제
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteLog(@PathVariable("id") long id) {
        logService.deleteOne(id);
        return ResponseEntity.noContent().build();
    }

    // 🆕 선택 삭제
    @PostMapping("/delete")
    public LogDeleteResponse deleteLogs(@RequestBody LogDeleteRequest req) {
        int deleted = logService.deleteMany(req.getIds());
        return LogDeleteResponse.builder().deletedCount(deleted).build();
    }

    // 🆕 내보내기
    @PostMapping("/export")
    public ResponseEntity<?> exportLogs(@RequestBody LogExportRequest req) {
        LogService.ExportResult result = logService.exportLogs(req);

        if (result.isJson()) {
            return ResponseEntity.ok(result.getJsonItems());
        }

        String csv = result.getCsvText();
        return ResponseEntity.ok()
                .header(HttpHeaders.CONTENT_TYPE, "text/csv; charset=UTF-8")
                .body(csv);
    }
}
