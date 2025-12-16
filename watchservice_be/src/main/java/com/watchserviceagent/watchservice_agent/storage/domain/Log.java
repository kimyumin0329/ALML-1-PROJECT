// src/main/java/com/watchserviceagent/watchservice_agent/storage/domain/Log.java
package com.watchserviceagent.watchservice_agent.storage.domain;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.Instant;

@Getter
@Builder
@ToString
public class Log {

    private final Long id;

    private final String ownerKey;
    private final String eventType;
    private final String path;

    private final boolean exists;
    private final long size;
    private final long lastModifiedTime;
    private final String hash;
    private final Double entropy;

    private final String aiLabel;
    private final Double aiScore;
    private final String aiDetail;

    private final Instant collectedAt;
}
