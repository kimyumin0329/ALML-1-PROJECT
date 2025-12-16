package com.watchserviceagent.watchservice_agent.alerts.dto;

import com.watchserviceagent.watchservice_agent.storage.dto.LogResponse;
import lombok.Builder;
import lombok.Getter;

import java.util.List;

@Getter
@Builder
public class AlertPageResponse {
    private final List<LogResponse> items;
    private final long total;
    private final int page;
    private final int size;
}
