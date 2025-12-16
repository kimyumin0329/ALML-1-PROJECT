package com.watchserviceagent.watchservice_agent.watcher.domain;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.nio.file.Path;
import java.time.Instant;

/**
 * WatchService 에서 발생한 "파일 시스템 이벤트"를 표현하는 도메인 객체.
 *
 * - ownerKey : SessionIdManager 에서 가져오는 사용자/세션 구분자
 * - eventType : CREATE / MODIFY / DELETE 중 하나
 * - fullPath  : 이벤트가 발생한 파일/디렉토리의 절대 경로
 * - eventTime : 이벤트 발생 시각 (백엔드 기준 시각)
 *
 * 이 객체는 Collector, 향후 EventWindowAggregator 가 공통으로 참조하는
 * "raw 이벤트" 기반이 된다.
 */
@Getter
@Builder
@ToString
public class WatcherEvent {

    /**
     * 세션/사용자 구분용 키.
     * 예: SessionIdManager.getSessionId() 값.
     */
    private final String ownerKey;

    /**
     * 이벤트 타입: "CREATE", "MODIFY", "DELETE"
     */
    private final String eventType;

    /**
     * 이벤트가 발생한 파일 또는 디렉토리의 절대 경로.
     */
    private final Path fullPath;

    /**
     * 이벤트 발생 시각 (서버 기준, UTC).
     */
    private final Instant eventTime;

    /**
     * 절대 경로를 문자열로 반환하는 편의 메서드.
     */
    public String getPathString() {
        return (fullPath != null) ? fullPath.toString() : null;
    }
}
