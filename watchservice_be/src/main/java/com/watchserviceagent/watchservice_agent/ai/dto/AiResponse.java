package com.watchserviceagent.watchservice_agent.ai.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;

/**
 * AI 서버에서 반환하는 JSON 응답을 매핑하는 DTO.
 *
 * 예시 응답(JSON 가정):
 * {
 *   "status": "ok",
 *   "label": "DANGER",
 *   "score": 0.92,
 *   "detail": "랜섬웨어 의심 패턴 감지"
 * }
 */
@Getter
@NoArgsConstructor
@ToString
public class AiResponse {

    // 응답 상태 (예: "ok", "error")
    private String status;

    // 분류 라벨 (예: "SAFE", "WARNING", "DANGER")
    private String label;

    // 위험 점수(0.0 ~ 1.0 등)
    private Double score;

    // 추가 설명/디버깅용 메시지
    private String detail;

    // 오류 메시지나 부가 메모(옵션)
    private String message;
}
