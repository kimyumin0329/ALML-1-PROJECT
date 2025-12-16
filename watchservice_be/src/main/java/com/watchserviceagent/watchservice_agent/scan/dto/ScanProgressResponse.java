package com.watchserviceagent.watchservice_agent.scan.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class ScanProgressResponse {
    private String status;      // RUNNING / PAUSED / DONE / ERROR
    private int percent;        // 0~100
    private long scanned;       // scanned file count
    private long total;         // total file count
    private String currentPath; // now processing
    private String message;     // optional
}
