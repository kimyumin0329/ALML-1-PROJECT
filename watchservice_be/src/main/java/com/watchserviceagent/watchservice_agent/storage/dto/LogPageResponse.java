package com.watchserviceagent.watchservice_agent.storage.dto;

import lombok.Builder;
import lombok.Getter;

import java.util.List;

@Getter
@Builder
public class LogPageResponse {
    private final List<LogResponse> items;
    private final long total;
    private final int page;
    private final int size;
}
