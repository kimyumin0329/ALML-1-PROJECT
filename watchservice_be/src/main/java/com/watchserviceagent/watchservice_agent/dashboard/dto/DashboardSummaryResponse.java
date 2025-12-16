package com.watchserviceagent.watchservice_agent.dashboard.dto;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

/**
 * 메인 대시보드 상단 요약 카드에 사용되는 응답 DTO.
 *
 * 프론트에서 사용하는 필드:
 *  - status      : "SAFE" / "WARNING" / "DANGER"
 *  - statusLabel : "안전" / "주의" / "위험"
 *  - lastEventTime : "YYYY-MM-DD HH:mm:ss" 또는 "-" (이벤트 없으면)
 */
@Getter
@Builder
@ToString
public class DashboardSummaryResponse {

    private final String status;       // SAFE / WARNING / DANGER
    private final String statusLabel;  // 안전 / 주의 / 위험
    private final String lastEventTime;

    private final int dangerCount;     // 최근 N개 중 DANGER 개수
    private final int warningCount;   // 최근 N개 중 WARNING 개수
    private final int totalCount;     // 최근 N개 전체 개수

    private final String watchedPath; // 현재 감시 중인 루트 경로 (선택)
}
