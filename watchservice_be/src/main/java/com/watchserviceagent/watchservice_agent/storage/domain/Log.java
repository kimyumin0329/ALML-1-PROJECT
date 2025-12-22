package com.watchserviceagent.watchservice_agent.storage.domain;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.Instant;

/**
 * 클래스 이름 : Log
 * 기능 : 데이터베이스에 저장되는 로그 엔티티. 파일 이벤트 정보와 AI 분석 결과를 포함한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
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

    // before/after 메타 및 diff
    private final Boolean existsBefore;
    private final Long sizeBefore;
    private final Long sizeAfter;
    private final Double entropyBefore;
    private final Double entropyAfter;
    private final String extBefore;
    private final String extAfter;
    private final Long sizeDiff;
    private final Double entropyDiff;
    private final long lastModifiedTime;
    private final String hash;
    private final Double entropy;

    private final String aiLabel;
    private final Double aiScore;
    private final String aiDetail;

    private final Instant collectedAt;
}
