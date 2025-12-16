package com.watchserviceagent.watchservice_agent.settings.dto;

import com.watchserviceagent.watchservice_agent.settings.domain.ExceptionRule;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

/**
 * 예외(화이트리스트) 규칙 응답 DTO.
 */
@Getter
@Builder
@ToString
public class ExceptionRuleResponse {

    private final Long id;
    private final String type;
    private final String pattern;
    private final String memo;
    private final String createdAt;

    private static final DateTimeFormatter DATE_TIME_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
                    .withZone(ZoneId.systemDefault());

    public static ExceptionRuleResponse from(ExceptionRule rule) {
        String createdAtStr = rule.getCreatedAt() != null
                ? DATE_TIME_FORMATTER.format(rule.getCreatedAt())
                : "-";

        return ExceptionRuleResponse.builder()
                .id(rule.getId())
                .type(rule.getType())
                .pattern(rule.getPattern())
                .memo(rule.getMemo())
                .createdAt(createdAtStr)
                .build();
    }
}
