package com.watchserviceagent.watchservice_agent.ai.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

@Getter
@Builder
@ToString
public class FamilyPredictRequest {

    /**
     * Python 서버가 기대하는 "features": {...}
     * - AiPayload를 그대로 넣어도 되고
     * - Map<String,Object>를 넣어도 됨
     */
    @JsonProperty("features")
    private Object features;

    @JsonProperty("topk")
    private Integer topk;
}
