package com.watchserviceagent.watchservice_agent.settings.domain;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.Instant;

/**
 * 감시 대상 폴더 설정 도메인 객체.
 *
 * - DB: watched_folder 테이블과 1:1 매핑.
 */
@Getter
@Builder
@ToString
public class WatchedFolder {

    private final Long id;          // PK (AUTOINCREMENT)
    private final String ownerKey;  // 설치/에이전트 구분용
    private final String name;      // 사용자에게 보일 이름
    private final String path;      // 실제 경로 (절대 경로 추천)
    private final Instant createdAt;
}
