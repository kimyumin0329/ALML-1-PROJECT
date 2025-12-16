package com.watchserviceagent.watchservice_agent.ai;

import com.watchserviceagent.watchservice_agent.ai.domain.AiResult;
import com.watchserviceagent.watchservice_agent.ai.dto.AiPayload;
import com.watchserviceagent.watchservice_agent.ai.dto.AiResponse;
import com.watchserviceagent.watchservice_agent.ai.dto.FamilyPredictRequest;
import com.watchserviceagent.watchservice_agent.ai.dto.FamilyPredictResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.*;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestClientException;
import org.springframework.web.client.RestTemplate;

import java.util.Collections;
import java.util.Map;

@Service
@RequiredArgsConstructor
@Slf4j
public class AiService {

    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.analyze.url:http://localhost:8000/api/analyze}")
    private String analyzeUrl;

    @Value("${ai.family.url:http://localhost:8001/predict}")
    private String familyUrl;

    public AiResult requestAnalysis(AiPayload payload) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            HttpEntity<AiPayload> entity = new HttpEntity<>(payload, headers);

            log.debug("[AiService] analyze 요청: url={}, payload={}", analyzeUrl, payload);

            AiResponse response = restTemplate.postForObject(
                    analyzeUrl,
                    entity,
                    AiResponse.class
            );

            log.debug("[AiService] analyze 응답: {}", response);
            return AiResult.fromResponse(response);

        } catch (RestClientException e) {
            log.error("[AiService] AI 서버 호출 실패", e);
            return AiResult.error("AI 서버 호출 실패: " + e.getMessage());
        } catch (Exception e) {
            log.error("[AiService] 예기치 못한 예외", e);
            return AiResult.error("AI 호출 중 예외: " + e.getMessage());
        }
    }

    /** ✅ (1) AiPayload 기반으로 family 분류 호출 */
    public FamilyPredictResponse requestFamilyPredict(AiPayload payload, int topk) {
        return requestFamilyPredictInternal(payload, topk);
    }

    /** ✅ (2) Map 기반으로 family 분류 호출 (추후 확장/테스트 편함) */
    public FamilyPredictResponse requestFamilyPredict(Map<String, Object> features, int topk) {
        return requestFamilyPredictInternal(features, topk);
    }

    private FamilyPredictResponse requestFamilyPredictInternal(Object features, int topk) {
        try {
            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_JSON);

            FamilyPredictRequest body = FamilyPredictRequest.builder()
                    .features(features)
                    .topk(topk)
                    .build();

            HttpEntity<FamilyPredictRequest> entity = new HttpEntity<>(body, headers);

            log.debug("[AiService] family 요청: url={}, body={}", familyUrl, body);

            FamilyPredictResponse response = restTemplate.postForObject(
                    familyUrl,
                    entity,
                    FamilyPredictResponse.class
            );

            log.debug("[AiService] family 응답: {}", response);
            return response;

        } catch (RestClientException e) {
            log.error("[AiService] Family 서버 호출 실패", e);
            return FamilyPredictResponse.builder()
                    .topk(Collections.emptyList())
                    .message("Family 서버 호출 실패: " + e.getMessage())
                    .build();
        } catch (Exception e) {
            log.error("[AiService] Family 호출 중 예외", e);
            return FamilyPredictResponse.builder()
                    .topk(Collections.emptyList())
                    .message("Family 호출 중 예외: " + e.getMessage())
                    .build();
        }
    }
}
