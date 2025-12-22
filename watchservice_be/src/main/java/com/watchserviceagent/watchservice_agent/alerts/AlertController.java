package com.watchserviceagent.watchservice_agent.alerts;

import com.watchserviceagent.watchservice_agent.alerts.dto.AlertPageResponse;
import com.watchserviceagent.watchservice_agent.alerts.dto.AlertStatsResponse;
import com.watchserviceagent.watchservice_agent.storage.dto.LogResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

/**
 * 클래스 이름 : AlertController
 * 기능 : 알림(위험 이벤트) 조회 및 통계를 제공하는 REST API 엔드포인트를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
@RestController
@RequestMapping("/alerts")
@RequiredArgsConstructor
public class AlertController {

    private final AlertService alertService;

    /**
     * 함수 이름 : getAlerts
     * 기능 : 페이지네이션, 필터링, 정렬을 지원하는 알림 목록을 조회한다.
     * 매개변수 : page - 페이지 번호 (1-based), size - 페이지 크기, from - 시작 날짜, to - 종료 날짜, level - 위험도 필터 (ALL|DANGER|WARNING|SAFE), keyword - 검색 키워드, sort - 정렬 기준
     * 반환값 : AlertPageResponse - 페이지네이션된 알림 목록
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping
    public AlertPageResponse getAlerts(
            @RequestParam(name = "page", required = false) Integer page,
            @RequestParam(name = "size", required = false) Integer size,
            @RequestParam(name = "from", required = false) String from,
            @RequestParam(name = "to", required = false) String to,
            @RequestParam(name = "level", required = false) String level,
            @RequestParam(name = "keyword", required = false) String keyword,
            @RequestParam(name = "sort", required = false) String sort
    ) {
        return alertService.getAlerts(page, size, from, to, level, keyword, sort);
    }

    /**
     * 함수 이름 : getAlert
     * 기능 : ID로 단일 알림의 상세 정보를 조회한다.
     * 매개변수 : id - 알림 ID
     * 반환값 : LogResponse - 알림 상세 정보
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/{id}")
    public LogResponse getAlert(@PathVariable("id") long id) {
        return alertService.getAlertById(id);
    }

    /**
     * 함수 이름 : stats
     * 기능 : 알림 통계를 일별 또는 주별로 조회한다.
     * 매개변수 : range - 통계 범위 (daily|weekly), from - 시작 날짜, to - 종료 날짜
     * 반환값 : AlertStatsResponse - 알림 통계 데이터
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/stats")
    public AlertStatsResponse stats(
            @RequestParam(name = "range", defaultValue = "daily") String range,
            @RequestParam(name = "from", required = false) String from,
            @RequestParam(name = "to", required = false) String to
    ) {
        return alertService.getStats(range, from, to);
    }
}
