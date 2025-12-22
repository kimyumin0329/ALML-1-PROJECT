package com.watchserviceagent.watchservice_agent.alerts.dto;

import lombok.Builder;
import lombok.Getter;

import java.util.List;

/**
 * 클래스 이름 : NotificationPageResponse
 * 기능 : 페이지네이션된 알림 목록 응답 DTO.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Getter
@Builder
public class NotificationPageResponse {
    private final List<NotificationResponse> items;
    private final long total;
    private final int page;
    private final int size;
}

