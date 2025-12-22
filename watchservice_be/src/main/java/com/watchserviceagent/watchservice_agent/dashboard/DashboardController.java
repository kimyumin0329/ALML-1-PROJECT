package com.watchserviceagent.watchservice_agent.dashboard;

import com.watchserviceagent.watchservice_agent.alerts.AlertService;
import com.watchserviceagent.watchservice_agent.alerts.dto.AlertPageResponse;
import com.watchserviceagent.watchservice_agent.dashboard.dto.DashboardSummaryResponse;
import com.watchserviceagent.watchservice_agent.settings.SettingsService;
import com.watchserviceagent.watchservice_agent.settings.dto.WatchedFolderResponse;
import com.watchserviceagent.watchservice_agent.storage.dto.LogResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

import java.util.List;

/**
 * 클래스 이름 : DashboardController
 * 기능 : 대시보드 요약 정보를 제공하는 REST API 엔드포인트를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@RestController
@RequestMapping("/dashboard")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
public class DashboardController {

    private final AlertService alertService;
    private final SettingsService settingsService;

    /**
     * 함수 이름 : getSummary
     * 기능 : 대시보드 요약 정보를 조회한다. 최근 알림, 위험도 상태, 감시 폴더 경로 등을 포함한다.
     * 매개변수 : 없음
     * 반환값 : DashboardSummaryResponse - 대시보드 요약 정보
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/summary")
    public DashboardSummaryResponse getSummary() {
        // 최근 알림(= ai_label 있는 로그) 50개 기준
        AlertPageResponse page = alertService.getAlerts(
                1, 50,
                null, null,
                "ALL",
                null,
                "collectedAt,desc"
        );

        List<LogResponse> items = page.getItems();
        int danger = 0;
        int warning = 0;

        for (LogResponse r : items) {
            if (r.getAiLabel() == null) continue;
            switch (r.getAiLabel().toUpperCase()) {
                case "DANGER" -> danger++;
                case "WARNING" -> warning++;
            }
        }

        String status;
        String statusLabel;

        if (danger > 0) {
            status = "DANGER";
            statusLabel = "위험";
        } else if (warning > 0) {
            status = "WARNING";
            statusLabel = "주의";
        } else {
            status = "SAFE";
            statusLabel = "안전";
        }

        String lastEventTime = "-";
        if (items != null && !items.isEmpty()) {
            lastEventTime = items.get(0).getCollectedAt(); // LogResponse가 이미 문자열로 내려줌
        }

        // watchedPath: 등록된 감시폴더 1개만 보여주기(여러 개면 첫 번째)
        String watchedPath = "-";
        try {
            List<WatchedFolderResponse> folders = settingsService.getWatchedFolders();
            if (folders != null && !folders.isEmpty()) watchedPath = folders.get(0).getPath();
        } catch (Exception ignore) {}

        DashboardSummaryResponse resp = DashboardSummaryResponse.builder()
                .status(status)
                .statusLabel(statusLabel)
                .lastEventTime(lastEventTime)
                .dangerCount(danger)
                .warningCount(warning)
                .totalCount(items == null ? 0 : items.size())
                .watchedPath(watchedPath)
                .build();

        log.info("[DashboardController] /dashboard/summary -> {}", resp);
        return resp;
    }
}
