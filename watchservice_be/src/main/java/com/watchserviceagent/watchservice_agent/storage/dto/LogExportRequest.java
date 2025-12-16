package com.watchserviceagent.watchservice_agent.storage.dto;

import lombok.Getter;

import java.util.List;

@Getter
public class LogExportRequest {
    private String format;      // "CSV" | "JSON"
    private List<Long> ids;     // 선택 내보내기 (비면 필터 기반 전체)
    private Filters filters;    // 필터 기반 전체

    @Getter
    public static class Filters {
        private Integer page;
        private Integer size;

        private String from;      // "YYYY-MM-DD"
        private String to;        // "YYYY-MM-DD"
        private String keyword;
        private String aiLabel;
        private String eventType;
        private String sort;      // "collectedAt,desc" 등
    }
}
