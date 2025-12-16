package com.watchserviceagent.watchservice_agent.storage.dto;

import lombok.Builder;
import lombok.Getter;

@Getter
@Builder
public class LogDeleteResponse {
    private final int deletedCount;
}
