package com.watchserviceagent.watchservice_agent.storage;

import com.watchserviceagent.watchservice_agent.storage.dto.*;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpHeaders;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 클래스 이름 : LogController
 * 기능 : 로그 조회, 삭제, 내보내기 등의 REST API 엔드포인트를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
@RestController
@RequestMapping("/logs")
@RequiredArgsConstructor
@Slf4j
public class LogController {

    private final LogService logService;

    /**
     * 함수 이름 : getRecentLogs
     * 기능 : 최근 로그를 지정된 개수만큼 조회한다.
     * 매개변수 : limit - 조회할 로그 개수 (기본값: 50, 최대: 1000)
     * 반환값 : LogResponse 리스트
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/recent")
    public List<LogResponse> getRecentLogs(@RequestParam(name = "limit", defaultValue = "50") int limit) {
        if (limit <= 0) limit = 50;
        else if (limit > 1000) limit = 1000;

        List<LogResponse> logs = logService.getRecentLogs(limit);
        log.info("[LogController] GET /logs/recent limit={} -> {}", limit, logs.size());
        return logs;
    }

    /**
     * 함수 이름 : getLogs
     * 기능 : 페이지네이션, 필터링, 정렬을 지원하는 로그 목록을 조회한다.
     * 매개변수 : page - 페이지 번호, size - 페이지 크기, from - 시작 날짜, to - 종료 날짜, keyword - 검색 키워드, aiLabel - AI 라벨 필터, eventType - 이벤트 타입 필터, sort - 정렬 기준
     * 반환값 : LogPageResponse - 페이지네이션된 로그 목록
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
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

    /**
     * 함수 이름 : getLog
     * 기능 : ID로 단일 로그의 상세 정보를 조회한다.
     * 매개변수 : id - 로그 ID
     * 반환값 : LogResponse - 로그 상세 정보
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/{id}")
    public LogResponse getLog(@PathVariable("id") long id) {
        return logService.getLogById(id);
    }

    /**
     * 함수 이름 : deleteLog
     * 기능 : 단일 로그를 삭제한다.
     * 매개변수 : id - 삭제할 로그 ID
     * 반환값 : ResponseEntity<Void> - 204 No Content
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @DeleteMapping("/{id}")
    public ResponseEntity<Void> deleteLog(@PathVariable("id") long id) {
        logService.deleteOne(id);
        return ResponseEntity.noContent().build();
    }

    /**
     * 함수 이름 : deleteLogs
     * 기능 : 여러 로그를 일괄 삭제한다.
     * 매개변수 : req - 삭제할 로그 ID 리스트를 포함한 요청 객체
     * 반환값 : LogDeleteResponse - 삭제된 로그 개수
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostMapping("/delete")
    public LogDeleteResponse deleteLogs(@RequestBody LogDeleteRequest req) {
        int deleted = logService.deleteMany(req.getIds());
        return LogDeleteResponse.builder().deletedCount(deleted).build();
    }

    /**
     * 함수 이름 : exportLogs
     * 기능 : 로그를 CSV 또는 JSON 형식으로 내보낸다.
     * 매개변수 : req - 내보내기 요청 (형식, 필터 조건 포함)
     * 반환값 : ResponseEntity - CSV 또는 JSON 형식의 로그 데이터
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
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
