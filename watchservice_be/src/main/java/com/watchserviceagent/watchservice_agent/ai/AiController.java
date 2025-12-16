package com.watchserviceagent.watchservice_agent.ai;

import com.watchserviceagent.watchservice_agent.ai.domain.AiResult;
import com.watchserviceagent.watchservice_agent.ai.dto.AiPayload;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;
import com.watchserviceagent.watchservice_agent.ai.dto.FamilyPredictResponse;
import java.util.Map;


/**
 * AI 서버 연동을 직접 테스트하기 위한 컨트롤러.
 *
 * ✔ 핵심 파이프라인(Watcher → Collector → Aggregator → AiService)에는 필수는 아님.
 * ✔ 프론트/포스트맨에서 직접 AiPayload를 보내서
 *    - AI 서버가 잘 동작하는지
 *    - 현재 모델이 어떤 라벨/점수를 주는지
 *   확인하는 용도의 디버깅/테스트 엔드포인트.
 */
@RestController
@RequestMapping("/ai")
@RequiredArgsConstructor
@Slf4j
public class AiController {

    private final AiService aiService; // AI 서버 호출 서비스

    /**
     * AI 분석 테스트용 엔드포인트.
     *
     * POST /ai/analyze
     * Body(JSON) 예시:
     * {
     *   "fileWriteCount": 14,
     *   "fileRenameCount": 3,
     *   "fileDeleteCount": 1,
     *   "fileModifyCount": 14,
     *   "fileEncryptLikeCount": 2,
     *   "changedFilesCount": 18,
     *   "entropyDiffMean": 0.92,
     *   "fileSizeDiffMean": 134.5,
     *   "randomExtensionFlag": 1
     * }
     */
    @PostMapping("/analyze")
    public AiResult analyze(@RequestBody AiPayload payload) {
        log.info("[AiController] /ai/analyze 요청 수신: {}", payload);

        AiResult result = aiService.requestAnalysis(payload);

        log.info("[AiController] /ai/analyze 결과: {}", result);
        return result;
    }

    /**
     * 랜섬웨어 패밀리 분류 테스트 엔드포인트(1): AiPayload 그대로 받기
     * POST /ai/family/predict?topk=5
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
     * 랜섬웨어 패밀리 분류 테스트 엔드포인트(2): Map으로 받기(유연)
     * POST /ai/family/predict/features?topk=5
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
     * 간단 헬스체크용 (원하면 프론트에서 ping 용도로 사용 가능).
     *
     * GET /ai/ping
     */
    @GetMapping("/ping")
    public String ping() {
        return "AI endpoint alive";
    }
}
