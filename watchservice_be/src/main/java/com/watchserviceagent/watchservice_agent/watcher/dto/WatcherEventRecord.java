package com.watchserviceagent.watchservice_agent.watcher.dto;

import com.watchserviceagent.watchservice_agent.watcher.domain.WatcherEvent;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.Instant;

/**
 * WatcherEvent 를 다른 계층(Collector, Storage, API)에 전달할 때 사용하는 DTO.
 *
 * - Path 대신 문자열 path 를 사용하고,
 * - 시간은 epochMillis(long)로 변환해서 전달한다.
 */
@Getter
@Builder
@ToString
public class WatcherEventRecord {

    private final String ownerKey;
    private final String eventType;   // CREATE / MODIFY / DELETE
    private final String path;        // 절대 경로 문자열
    private final long eventTimeMs;   // epoch millis

    public static WatcherEventRecord from(WatcherEvent event) {
        Instant t = event.getEventTime();
        long epochMillis = (t != null) ? t.toEpochMilli() : 0L;

        return WatcherEventRecord.builder()
                .ownerKey(event.getOwnerKey())
                .eventType(event.getEventType())
                .path(event.getPathString())
                .eventTimeMs(epochMillis)
                .build();
    }
}
