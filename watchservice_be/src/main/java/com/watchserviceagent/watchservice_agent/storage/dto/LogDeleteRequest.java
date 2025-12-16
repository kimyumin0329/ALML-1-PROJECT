package com.watchserviceagent.watchservice_agent.storage.dto;

import lombok.Getter;

import java.util.List;

@Getter
public class LogDeleteRequest {
    private List<Long> ids;
}
