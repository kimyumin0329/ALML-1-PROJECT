package com.watchserviceagent.watchservice_agent.scan.dto;

import lombok.Getter;
import lombok.ToString;

import java.util.List;

@Getter
@ToString
public class ScanStartRequest {

    // POST /scan/start  Body: { paths:[...] }
    private List<String> paths;

    // default true (null이면 true로 처리)
    private Boolean autoStartWatcher;
}
