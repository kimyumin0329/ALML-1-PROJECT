package com.watchserviceagent.watchservice_agent.ai;

import com.watchserviceagent.watchservice_agent.ai.domain.AiResult;
import com.watchserviceagent.watchservice_agent.ai.dto.AiPayload;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import com.watchserviceagent.watchservice_agent.ai.dto.FamilyPredictResponse;
import java.util.Map;


/**
 * 클래스 이름 : AiController
 * 기능 : AI 서버 연동을 직접 테스트하기 위한 REST API 엔드포인트를 제공한다. 디버깅 및 테스트 용도.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@RestController
@RequestMapping("/ai")
@RequiredArgsConstructor
@Slf4j
public class AiController {

    private final AiService aiService;

    /**
     * 함수 이름 : analyze
     * 기능 : AI 서버에 행위 분석 요청을 보내고 결과를 반환한다. 테스트용 엔드포인트.
     * 매개변수 : payload - 윈도우 집계된 피처 벡터
     * 반환값 : AiResult - AI 분석 결과
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostMapping("/analyze")
    public AiResult analyze(@RequestBody AiPayload payload) {
        log.info("[AiController] /ai/analyze 요청 수신: {}", payload);

        AiResult result = aiService.requestAnalysis(payload);

        log.info("[AiController] /ai/analyze 결과: {}", result);
        return result;
    }

    /**
     * 함수 이름 : predictFamily
     * 기능 : AiPayload를 기반으로 랜섬웨어 패밀리 분류를 요청한다. 테스트용 엔드포인트.
     * 매개변수 : payload - 피처 벡터, topk - 상위 k개 결과 반환 개수
     * 반환값 : FamilyPredictResponse - 패밀리 분류 결과
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostMapping("/family/predict")
    public FamilyPredictResponse predictFamily(
            @RequestBody AiPayload payload,
            @RequestParam(defaultValue = "5") int topk
    ) {
        log.info("[AiController] /ai/family/predict 요청: payload={}, topk={}", payload, topk);
        FamilyPredictResponse res = aiService.requestFamilyPredict(payload, topk);
        log.info("[AiController] /ai/family/predict 응답: {}", res);
        return res;
    }

    /**
     * 함수 이름 : predictFamilyByFeatures
     * 기능 : Map 형태의 피처를 기반으로 랜섬웨어 패밀리 분류를 요청한다. 테스트용 엔드포인트.
     * 매개변수 : features - 피처 맵, topk - 상위 k개 결과 반환 개수
     * 반환값 : FamilyPredictResponse - 패밀리 분류 결과
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostMapping("/family/predict/features")
    public FamilyPredictResponse predictFamilyByFeatures(
            @RequestBody Map<String, Object> features,
            @RequestParam(defaultValue = "5") int topk
    ) {
        log.info("[AiController] /ai/family/predict/features 요청: topk={}, features={}", topk, features);
        FamilyPredictResponse res = aiService.requestFamilyPredict(features, topk);
        log.info("[AiController] /ai/family/predict/features 응답: {}", res);
        return res;
    }



    /**
     * 함수 이름 : ping
     * 기능 : AI 엔드포인트의 헬스체크를 수행한다.
     * 매개변수 : 없음
     * 반환값 : String - "AI endpoint alive"
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/ping")
    public String ping() {
        return "AI endpoint alive";
    }
}
