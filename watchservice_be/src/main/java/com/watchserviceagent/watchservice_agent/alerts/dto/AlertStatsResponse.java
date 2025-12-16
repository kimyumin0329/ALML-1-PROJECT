package com.watchserviceagent.watchservice_agent.alerts.dto;

import lombok.Builder;
import lombok.Getter;

import java.util.List;

@Getter
@Builder
public class AlertStatsResponse {
    private final String range; // daily|weekly
    private final String from;
    private final String to;
    private final List<SeriesPoint> series;

    @Getter
    @Builder
    public static class SeriesPoint {
        private final String date; // daily: YYYY-MM-DD, weekly: YYYY-W##
        private final int warning;
        private final int danger;
    }
}
