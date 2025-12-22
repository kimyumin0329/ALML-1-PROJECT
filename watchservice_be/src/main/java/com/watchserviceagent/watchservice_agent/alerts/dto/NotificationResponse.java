package com.watchserviceagent.watchservice_agent.alerts.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.util.List;

/**
 * 클래스 이름 : NotificationResponse
 * 기능 : 프론트엔드로 전달하는 알림 응답 DTO. 윈도우 단위 AI 분석 결과와 영향 받은 파일 경로 목록을 포함한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Getter
@Builder
@ToString
public class NotificationResponse {

    private final long id;

    private final String windowStart;    // 문자열로 포맷팅
    private final String windowEnd;      // 문자열로 포맷팅
    private final String createdAt;      // 문자열로 포맷팅

    // AI 분석 결과
    private final String aiLabel;        // SAFE / WARNING / DANGER / UNKNOWN
    private final Double aiScore;        // 위험도 (0~1)
    private final String topFamily;      // Benign / LockBit / ...
    private final String aiDetail;       // 상세 메시지

    // 영향 받은 파일 정보
    private final int affectedFilesCount;
    private final List<String> affectedPaths;  // 전체 경로 리스트
}

