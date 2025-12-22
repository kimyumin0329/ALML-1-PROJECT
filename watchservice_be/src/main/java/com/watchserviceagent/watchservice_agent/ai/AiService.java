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

/**
 * 클래스 이름 : AiService
 * 기능 : 외부 AI 서버(ransomware-family-classifier)와 통신하여 행위 분석 및 랜섬웨어 패밀리 분류를 수행한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class AiService {

    private final RestTemplate restTemplate = new RestTemplate();

    @Value("${ai.analyze.url:http://localhost:8000/api/analyze}")
    private String analyzeUrl;

    @Value("${ai.family.url:http://localhost:8001/predict}")
    private String familyUrl;

    /**
     * 함수 이름 : requestAnalysis
     * 기능 : AI 서버에 행위 분석 요청을 보내고 결과를 반환한다. 랜섬웨어 감지 시 경고 로그를 출력한다.
     * 매개변수 : payload - 윈도우 집계된 피처 벡터
     * 반환값 : AiResult - AI 분석 결과 (라벨, 점수, topFamily, 랜섬웨어 여부 포함)
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
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
            
            AiResult result = AiResult.fromResponse(response);
            
            // 랜섬웨어 감지 시 경고 로그
            if (result.getIsRansomware() != null && result.getIsRansomware()) {
                log.warn("[AiService] ⚠️ 랜섬웨어 감지! topFamily={}, score={}, label={}", 
                        result.getTopFamily(), result.getScore(), result.getLabel());
            } else {
                log.info("[AiService] AI 분석 완료. topFamily={}, isRansomware={}, score={}, label={}", 
                        result.getTopFamily(), result.getIsRansomware(), result.getScore(), result.getLabel());
            }
            
            return result;

        } catch (RestClientException e) {
            log.error("[AiService] AI 서버 호출 실패", e);
            return AiResult.error("AI 서버 호출 실패: " + e.getMessage());
        } catch (Exception e) {
            log.error("[AiService] 예기치 못한 예외", e);
            return AiResult.error("AI 호출 중 예외: " + e.getMessage());
        }
    }

    /**
     * 함수 이름 : requestFamilyPredict
     * 기능 : AiPayload를 기반으로 랜섬웨어 패밀리 분류를 요청한다.
     * 매개변수 : payload - 피처 벡터, topk - 상위 k개 결과 반환 개수
     * 반환값 : FamilyPredictResponse - 패밀리 분류 결과 (topFamily, category, verdict 포함)
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public FamilyPredictResponse requestFamilyPredict(AiPayload payload, int topk) {
        return requestFamilyPredictInternal(payload, topk);
    }

    /**
     * 함수 이름 : requestFamilyPredict
     * 기능 : Map 형태의 피처를 기반으로 랜섬웨어 패밀리 분류를 요청한다.
     * 매개변수 : features - 피처 맵, topk - 상위 k개 결과 반환 개수
     * 반환값 : FamilyPredictResponse - 패밀리 분류 결과
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public FamilyPredictResponse requestFamilyPredict(Map<String, Object> features, int topk) {
        return requestFamilyPredictInternal(features, topk);
    }

    /**
     * 함수 이름 : requestFamilyPredictInternal
     * 기능 : AI 서버에 패밀리 분류 요청을 보내고, 결과를 후처리하여 카테고리와 랜섬웨어 여부를 판정한다.
     * 매개변수 : features - 피처 객체 (AiPayload 또는 Map), topk - 상위 k개 결과 반환 개수
     * 반환값 : FamilyPredictResponse - 후처리된 패밀리 분류 결과
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
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

            FamilyPredictResponse raw = restTemplate.postForObject(
                    familyUrl,
                    entity,
                    FamilyPredictResponse.class
            );

            log.debug("[AiService] family 응답: {}", raw);

            // === 후처리: top-1 family 기준으로 카테고리/랜섬웨어 여부 판정 ===
            String topFamily = null;
            Double topProb = null;
            if (raw != null && raw.getTopk() != null && !raw.getTopk().isEmpty()) {
                FamilyPredictResponse.Item top1 = raw.getTopk().get(0);
                topFamily = top1.getFamily();
                topProb = top1.getProb();
            }

            String category = classifyFamilyCategory(topFamily);
            boolean isRansomware = "RANSOMWARE".equals(category);
            String verdict;
            if (topFamily == null || topFamily.isBlank()) {
                verdict = "분류 불가";
            } else if ("BENIGN".equals(category)) {
                verdict = "정상";
            } else if (isRansomware) {
                verdict = "랜섬웨어";
            } else {
                verdict = "악성(기타)";
            }

            return FamilyPredictResponse.builder()
                    .topk(raw != null ? raw.getTopk() : Collections.emptyList())
                    .message(raw != null ? raw.getMessage() : null)
                    .topFamily(topFamily)
                    .category(category)
                    .ransomware(isRansomware)
                    .verdict(verdict)
                    .build();

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

    /**
     * 함수 이름 : classifyFamilyCategory
     * 기능 : 랜섬웨어 패밀리 이름을 기반으로 상위 카테고리를 판정한다.
     * 매개변수 : family - 패밀리 이름
     * 반환값 : "BENIGN", "RANSOMWARE", "INFOSTEALER", "RAT", "BOTNET_LOADER", "OTHER" 중 하나
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private String classifyFamilyCategory(String family) {
        if (family == null || family.isBlank()) {
            return "OTHER";
        }
        String f = family.trim();

        // 정상
        if ("Benign".equalsIgnoreCase(f)) {
            return "BENIGN";
        }

        // 랜섬웨어
        if (matchesAny(f,
                "Cerber", "DarkSide", "Dharma", "Gandcrab", "LockBit",
                "Maze", "Phobos", "REvil", "Ragnar", "Ryuk",
                "Shade", "WannaCry")) {
            return "RANSOMWARE";
        }

        // 인포스틸러 / 크리덴셜 탈취
        if (matchesAny(f,
                "Raccoon", "RedLine", "Snake")) {
            return "INFOSTEALER";
        }

        // RAT
        if (matchesAny(f,
                "Agenttesla", "Gh0st", "NanoCore", "njRat", "Remcos")) {
            return "RAT";
        }

        // 봇넷 / 로더
        if (matchesAny(f,
                "Emotet", "Qbot", "Ursnif", "Glupteba", "Guloader", "Formbook")) {
            return "BOTNET_LOADER";
        }

        return "OTHER";
    }

    /**
     * 함수 이름 : matchesAny
     * 기능 : 주어진 값이 후보 문자열 중 하나와 대소문자 무시하고 일치하는지 확인한다.
     * 매개변수 : value - 확인할 문자열, candidates - 후보 문자열들
     * 반환값 : 일치하면 true, 아니면 false
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private boolean matchesAny(String value, String... candidates) {
        for (String c : candidates) {
            if (c.equalsIgnoreCase(value)) {
                return true;
            }
        }
        return false;
    }
}
