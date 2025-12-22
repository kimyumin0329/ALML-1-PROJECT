package com.watchserviceagent.watchservice_agent.ai.dto;

import lombok.*;

import java.util.List;

@Getter
@Builder
@NoArgsConstructor
@AllArgsConstructor
@ToString
public class FamilyPredictResponse {

    private List<Item> topk;
    private String message;

    // === 백엔드에서 후처리로 채우는 편의 필드들 ===
    // top-k 중 가장 높은 확률의 family (예: "Benign", "LockBit")
    private String topFamily;
    // 상위 타입: BENIGN / RANSOMWARE / INFOSTEALER / RAT / BOTNET_LOADER / OTHER
    private String category;
    // 랜섬웨어 여부: true면 랜섬웨어 계열로 판단
    private Boolean ransomware;
    // 프론트 표시용 한글 요약: "정상", "랜섬웨어", "악성(기타)" 등
    private String verdict;

    @Getter
    @NoArgsConstructor
    @AllArgsConstructor
    @ToString
    public static class Item {
        private String family;
        private Double prob;
    }
}
