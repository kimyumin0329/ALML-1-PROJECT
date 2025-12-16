package com.watchserviceagent.watchservice_agent.scan.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class ScanStartResponse {
    private String scanId;
}
