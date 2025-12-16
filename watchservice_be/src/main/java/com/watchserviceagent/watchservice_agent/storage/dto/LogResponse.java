package com.watchserviceagent.watchservice_agent.storage.dto;

import com.watchserviceagent.watchservice_agent.storage.domain.Log;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

/**
 * 프론트엔드로 내려보내는 로그 응답 DTO.
 *
 * - 프론트는 이 필드들을 그대로 테이블/카드 렌더링에 사용한다.
 */
@Getter
@ToString
@Builder
public class LogResponse {

    private final long id;

    private final String collectedAt; // 문자열로 포맷팅
    private final String eventType;
    private final String path;

    private final boolean exists;
    private final long size;

    private final String lastModifiedTime; // 문자열로 포맷팅
    private final String hash;
    private final Double entropy;

    private final String aiLabel;   // SAFE/WARNING/DANGER
    private final Double aiScore;
    private final String aiDetail;

    private static final DateTimeFormatter DATE_TIME_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault());

    public static LogResponse from(Log log) {
        if (log == null) return null;

        String collectedAtStr = log.getCollectedAt() == null
                ? null
                : DATE_TIME_FORMATTER.format(log.getCollectedAt());

        String lastModifiedStr = DATE_TIME_FORMATTER.format(
                Instant.ofEpochMilli(log.getLastModifiedTime())
        );

        return LogResponse.builder()
                .id(log.getId())
                .collectedAt(collectedAtStr)
                .eventType(log.getEventType())
                .path(log.getPath())
                .exists(log.isExists())
                .size(log.getSize())
                .lastModifiedTime(lastModifiedStr)
                .hash(log.getHash())
                .entropy(log.getEntropy())
                .aiLabel(log.getAiLabel())
                .aiScore(log.getAiScore())
                .aiDetail(log.getAiDetail())
                .build();
    }
}
