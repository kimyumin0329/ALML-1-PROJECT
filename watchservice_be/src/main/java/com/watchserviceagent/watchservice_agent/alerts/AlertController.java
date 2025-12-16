package com.watchserviceagent.watchservice_agent.alerts;

import com.watchserviceagent.watchservice_agent.alerts.dto.AlertPageResponse;
import com.watchserviceagent.watchservice_agent.alerts.dto.AlertStatsResponse;
import com.watchserviceagent.watchservice_agent.storage.dto.LogResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
@RestController
@RequestMapping("/alerts")
@RequiredArgsConstructor
public class AlertController {

    private final AlertService alertService;

    // GET /alerts?page=1&size=50&from=YYYY-MM-DD&to=YYYY-MM-DD&level=&keyword=&sort=
    @GetMapping
    public AlertPageResponse getAlerts(
            @RequestParam(name = "page", required = false) Integer page,   // 1-based 권장(프론트는 0-based면 +1해서 보냄)
            @RequestParam(name = "size", required = false) Integer size,
            @RequestParam(name = "from", required = false) String from,
            @RequestParam(name = "to", required = false) String to,
            @RequestParam(name = "level", required = false) String level,   // ALL|DANGER|WARNING|SAFE (ALL/null이면 전체)
            @RequestParam(name = "keyword", required = false) String keyword,
            @RequestParam(name = "sort", required = false) String sort
    ) {
        return alertService.getAlerts(page, size, from, to, level, keyword, sort);
    }

    // GET /alerts/{id}
    @GetMapping("/{id}")
    public LogResponse getAlert(@PathVariable("id") long id) {
        return alertService.getAlertById(id);
    }

    // GET /alerts/stats?range=daily|weekly&from=&to=
    @GetMapping("/stats")
    public AlertStatsResponse stats(
            @RequestParam(name = "range", defaultValue = "daily") String range,
            @RequestParam(name = "from", required = false) String from,
            @RequestParam(name = "to", required = false) String to
    ) {
        return alertService.getStats(range, from, to);
    }
}
