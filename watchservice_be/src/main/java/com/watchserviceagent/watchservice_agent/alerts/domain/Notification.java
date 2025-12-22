package com.watchserviceagent.watchservice_agent.alerts.domain;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.Instant;
import java.util.List;

/**
 * 클래스 이름 : Notification
 * 기능 : 윈도우 단위 AI 분석 결과를 저장하는 도메인 엔티티. 영향 받은 파일 경로 목록을 포함한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Getter
@Builder
@ToString
public class Notification {

    private final Long id;
    private final String ownerKey;
    private final Instant windowStart;
    private final Instant windowEnd;
    private final Instant createdAt;

    // AI 분석 결과
    private final String aiLabel;        // SAFE / WARNING / DANGER / UNKNOWN
    private final Double aiScore;        // 위험도 (0~1)
    private final String topFamily;      // Benign / LockBit / ...
    private final String aiDetail;       // 상세 메시지

    // 영향 받은 파일 정보
    private final int affectedFilesCount;
    private final List<String> affectedPaths;  // 전체 경로 리스트
}

