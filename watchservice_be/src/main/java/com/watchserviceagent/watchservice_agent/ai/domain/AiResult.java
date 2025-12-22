package com.watchserviceagent.watchservice_agent.ai.domain;

import com.watchserviceagent.watchservice_agent.ai.dto.AiResponse;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

/**
 * 클래스 이름 : AiResult
 * 기능 : AI 분석 결과를 내부에서 사용하기 위한 도메인 객체. 라벨, 점수, topFamily, 랜섬웨어 여부를 포함한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Getter
@Builder
@ToString
public class AiResult {

    private String label;        // SAFE / WARNING / DANGER / UNKNOWN
    private Double score;        // "위험도"로 저장 (0~1)
    private String detail;       // 로그용
    private String topFamily;    // Benign / LockBit ...
    private Boolean isRansomware;

    /**
     * 함수 이름 : fromResponse
     * 기능 : AiResponse DTO를 AiResult 도메인 객체로 변환한다. topFamily를 기반으로 라벨과 위험도를 결정한다.
     * 매개변수 : response - AI 서버 응답 DTO
     * 반환값 : AiResult - 변환된 도메인 객체
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public static AiResult fromResponse(AiResponse response) {
        if (response == null) {
            return error("AI 응답 없음(null)");
        }

        String topFamily = response.getTopFamily();
        Double topProb = response.getTopProb();
        boolean isRansomware = response.isRansomware();

        // ✅ Benign이면 무조건 SAFE (AI가 DANGER로 줘도 덮어씀)
        String label;
        Double riskScore = null;

        if (topFamily == null || topFamily.isBlank()) {
            label = "UNKNOWN";
        } else if ("Benign".equalsIgnoreCase(topFamily.trim())) {
            label = "SAFE";
            if (topProb != null) riskScore = clamp01(1.0 - topProb); // benign 확률 -> 위험도 변환
        } else {
            // 악성 계열이면 최소 WARNING, 확신 높으면 DANGER
            if (topProb != null) {
                riskScore = clamp01(topProb);
                label = (riskScore >= 0.70) ? "DANGER" : "WARNING";
            } else {
                label = "WARNING";
            }
        }

        String detailStr = buildDetail(topFamily, topProb, response.getMessage(), response.getDetail());

        return AiResult.builder()
                .label(label)
                .score(riskScore)
                .detail(detailStr)
                .topFamily(topFamily)
                .isRansomware(isRansomware)
                .build();
    }

    private static String buildDetail(String topFamily, Double topProb, String message, String legacyDetail) {
        StringBuilder sb = new StringBuilder();
        if (topFamily != null) sb.append("top_family=").append(topFamily);
        if (topProb != null) sb.append(", top_prob=").append(topProb);
        if (message != null && !message.isBlank()) sb.append(", message=").append(message);
        // legacy detail도 남기고 싶으면(옵션)
        if (legacyDetail != null && !legacyDetail.isBlank()) sb.append(", detail=").append(legacyDetail);
        if (sb.isEmpty()) return "AI 분석 완료";
        return sb.toString();
    }

    private static double clamp01(double v) {
        if (v < 0) return 0;
        if (v > 1) return 1;
        return v;
    }

    /**
     * 함수 이름 : error
     * 기능 : 에러 상황에서 사용할 AiResult를 생성한다.
     * 매개변수 : message - 에러 메시지
     * 반환값 : AiResult - 에러 상태의 AiResult (label=UNKNOWN, isRansomware=false)
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public static AiResult error(String message) {
        return AiResult.builder()
                .label("UNKNOWN")
                .score(null)
                .detail(message)
                .topFamily(null)
                .isRansomware(false)
                .build();
    }
}
