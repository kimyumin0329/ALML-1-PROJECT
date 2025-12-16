package com.watchserviceagent.watchservice_agent.alerts;

import com.watchserviceagent.watchservice_agent.alerts.dto.AlertPageResponse;
import com.watchserviceagent.watchservice_agent.alerts.dto.AlertStatsResponse;
import com.watchserviceagent.watchservice_agent.common.util.SessionIdManager;
import com.watchserviceagent.watchservice_agent.storage.LogRepository;
import com.watchserviceagent.watchservice_agent.storage.domain.Log;
import com.watchserviceagent.watchservice_agent.storage.dto.LogResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.List;
import java.util.Locale;
import java.util.NoSuchElementException;

@Service
@RequiredArgsConstructor
public class AlertService {

    private final SessionIdManager sessionIdManager;
    private final LogRepository logRepository;

    private static final int MAX_PAGE_SIZE = 1000;
    private static final DateTimeFormatter DATE_TIME_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    public AlertPageResponse getAlerts(
            Integer page,
            Integer size,
            String from,
            String to,
            String level,
            String keyword,
            String sort
    ) {
        int p = (page == null || page < 1) ? 1 : page;
        int s = (size == null || size < 1) ? 50 : Math.min(size, MAX_PAGE_SIZE);

        SortInfo si = parseSort(sort);

        Long fromEpoch = parseFromToEpochStart(from);
        Long toEpoch = parseFromToEpochEnd(to);

        String ownerKey = sessionIdManager.getSessionId();

        // level 정규화: ALL/빈값 => null 처리
        String lv = normalizeLevel(level);

        long total = logRepository.countAlertsByOwner(ownerKey, fromEpoch, toEpoch, keyword, lv);

        int offset = (p - 1) * s;
        List<Log> logs = logRepository.findAlertsByOwner(
                ownerKey, fromEpoch, toEpoch, keyword, lv,
                si.field, si.dir, offset, s
        );

        List<LogResponse> items = logs.stream().map(LogResponse::from).toList();

        return AlertPageResponse.builder()
                .items(items)
                .total(total)
                .page(p)
                .size(s)
                .build();
    }

    public LogResponse getAlertById(long id) {
        String ownerKey = sessionIdManager.getSessionId();
        Log logEntity = logRepository.findAlertByIdAndOwner(ownerKey, id)
                .orElseThrow(() -> new NoSuchElementException("alert not found: id=" + id));
        return LogResponse.from(logEntity);
    }

    public AlertStatsResponse getStats(String range, String from, String to) {
        String rg = (range == null) ? "daily" : range.trim().toLowerCase(Locale.ROOT);
        if (!rg.equals("daily") && !rg.equals("weekly")) rg = "daily";

        Long fromEpoch = parseFromToEpochStart(from);
        Long toEpoch = parseFromToEpochEnd(to);

        String ownerKey = sessionIdManager.getSessionId();

        var rows = logRepository.getAlertStats(ownerKey, fromEpoch, toEpoch, rg);

        var series = rows.stream()
                .map(r -> AlertStatsResponse.SeriesPoint.builder()
                        .date(r.bucket())
                        .warning((int) r.warning())
                        .danger((int) r.danger())
                        .build())
                .toList();

        return AlertStatsResponse.builder()
                .range(rg)
                .from(from)
                .to(to)
                .series(series)
                .build();
    }

    private String normalizeLevel(String level) {
        if (level == null) return null;
        String v = level.trim().toUpperCase(Locale.ROOT);
        if (v.isBlank() || v.equals("ALL")) return null;
        if (v.equals("DANGER") || v.equals("WARNING") || v.equals("SAFE")) return v;
        return null;
    }

    private SortInfo parseSort(String sort) {
        if (sort == null || sort.isBlank()) return new SortInfo("collectedAt", "desc");
        String[] parts = sort.split(",");
        String field = parts.length > 0 ? parts[0].trim() : "collectedAt";
        String dir = parts.length > 1 ? parts[1].trim() : "desc";
        if (field.isBlank()) field = "collectedAt";
        if (dir.isBlank()) dir = "desc";
        return new SortInfo(field, dir);
    }

    private Long parseFromToEpochStart(String from) {
        if (from == null || from.isBlank()) return null;
        try {
            LocalDate d = LocalDate.parse(from.trim());
            return d.atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (DateTimeParseException ignore) {}
        try {
            LocalDateTime dt = LocalDateTime.parse(from.trim(), DATE_TIME_FMT);
            return dt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (Exception ignore) {}
        return null;
    }

    private Long parseFromToEpochEnd(String to) {
        if (to == null || to.isBlank()) return null;
        try {
            LocalDate d = LocalDate.parse(to.trim());
            // inclusive end-of-day
            LocalDateTime end = d.atTime(23, 59, 59, 999_000_000);
            return end.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (DateTimeParseException ignore) {}
        try {
            LocalDateTime dt = LocalDateTime.parse(to.trim(), DATE_TIME_FMT);
            return dt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (Exception ignore) {}
        return null;
    }

    private static class SortInfo {
        final String field;
        final String dir;
        SortInfo(String field, String dir) { this.field = field; this.dir = dir; }
    }
}
