package com.watchserviceagent.watchservice_agent.ai.domain;

import com.watchserviceagent.watchservice_agent.ai.dto.AiResponse;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

/**
 * AI 분석 결과를 내부에서 사용하기 위한 도메인 객체.
 *
 * - label : "SAFE" / "WARNING" / "DANGER" / "UNKNOWN"
 * - score : 0.0 ~ 1.0 (위험도)
 * - detail: 설명/로그용 메시지
 */
@Getter
@Builder
@ToString
public class AiResult {

    private String label;
    private Double score;
    private String detail;

    /**
     * AiResponse DTO를 AiResult 도메인으로 변환.
     * 응답이 null 이거나 status 가 error 이면 UNKNOWN 으로 처리.
     */
    public static AiResult fromResponse(AiResponse response) {
        if (response == null) {
            return AiResult.builder()
                    .label("UNKNOWN")
                    .score(null)
                    .detail("AI 응답 없음(null)")
                    .build();
        }

        String label = response.getLabel();
        if (label == null || label.isBlank()) {
            label = "UNKNOWN";
        }

        return AiResult.builder()
                .label(label)
                .score(response.getScore())
                .detail(response.getDetail() != null ? response.getDetail() : response.getMessage())
                .build();
    }

    /**
     * 예외/오류 상황에서 사용할 헬퍼.
     */
    public static AiResult error(String message) {
        return AiResult.builder()
                .label("UNKNOWN")
                .score(null)
                .detail(message)
                .build();
    }
}
