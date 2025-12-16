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

    @Getter
    @NoArgsConstructor
    @AllArgsConstructor
    @ToString
    public static class Item {
        private String family;
        private Double prob;
    }
}
