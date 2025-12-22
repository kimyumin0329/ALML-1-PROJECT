package com.watchserviceagent.watchservice_agent.ai.dto;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.ToString;

import java.util.List;

/**
 * AI 서버 응답 DTO (두 형식 모두 수용)
 *
 * (A) analyze 형식(기존):
 * {
 *   "status": "ok",
 *   "label": "DANGER" | "WARNING" | "SAFE",
 *   "score": 0.92,
 *   "detail": "top_family=LockBit",
 *   "message": "..."
 * }
 *
 * (B) family/predict 형식(현재 네가 보여준 것):
 * {
 *   "topk":[{"family":"Benign","prob":0.99}, ...],
 *   "message":"Missing ..."
 * }
 */
@Getter
@NoArgsConstructor
@ToString
public class AiResponse {

    private String status;

    // (A) 형식
    private String label;
    private Double score;
    private String detail;
    private String message;

    // (B) 형식
    private List<TopK> topk;

    @Getter
    @NoArgsConstructor
    @ToString
    public static class TopK {
        private String family;
        private Double prob;
    }

    /** top_family 추출: detail 우선, 없으면 topk[0].family */
    public String getTopFamily() {
        // 1) detail에서 파싱
        if (detail != null && !detail.isBlank()) {
            String prefix = "top_family=";
            int idx = detail.indexOf(prefix);
            if (idx >= 0) {
                String after = detail.substring(idx + prefix.length()).trim();
                int endIdx = after.indexOf(',');
                if (endIdx > 0) after = after.substring(0, endIdx).trim();
                if (!after.isBlank()) return after;
            }
        }

        // 2) topk에서 추출
        if (topk != null && !topk.isEmpty()) {
            String fam = topk.get(0).getFamily();
            if (fam != null && !fam.isBlank()) return fam.trim();
        }
        return null;
    }

    /** top 확률 추출: score 우선, 없으면 topk[0].prob */
    public Double getTopProb() {
        if (score != null) return score;
        if (topk != null && !topk.isEmpty()) {
            return topk.get(0).getProb();
        }
        return null;
    }

    /** Benign이면 false, 그 외는 true(가정) */
    public boolean isRansomware() {
        String topFamily = getTopFamily();
        if (topFamily == null || topFamily.isBlank()) return false;
        return !"Benign".equalsIgnoreCase(topFamily.trim());
    }
}
