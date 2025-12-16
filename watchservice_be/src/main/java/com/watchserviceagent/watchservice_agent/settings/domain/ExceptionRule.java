package com.watchserviceagent.watchservice_agent.settings.domain;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.Instant;

/**
 * 예외(화이트리스트) 규칙 도메인 객체.
 *
 * - type: PATH / EXT / 기타
 * - pattern: 경로 전체 또는 확장자 패턴 등
 */
@Getter
@Builder
@ToString
public class ExceptionRule {

    private final Long id;          // PK (AUTOINCREMENT)
    private final String ownerKey;  // 설치/에이전트 구분용
    private final String type;      // PATH / EXT / ...
    private final String pattern;   // 경로 또는 패턴
    private final String memo;      // 메모/설명
    private final Instant createdAt;
}
