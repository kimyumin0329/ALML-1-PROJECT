package com.watchserviceagent.watchservice_agent.collector.dto;

import com.watchserviceagent.watchservice_agent.ai.domain.AiResult;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.Instant;

/**
 * "이 파일에 대해 이 이벤트 1번이 발생했을 때,
 *  이전 상태 vs 현재 상태를 비교한 결과 한 건"을 나타내는 DTO.
 *
 * Collector → Analytics/EventWindowAggregator → Storage/LogWriterWorker 에서 공통 사용.
 */
@Getter
@Builder
@ToString
public class FileAnalysisResult {

    // ===== 공통 메타 정보 =====

    // 설치/에이전트 식별용 ownerKey
    private final String ownerKey;

    // 이벤트 타입 (CREATE / MODIFY / DELETE)
    private final String eventType;

    // 파일 절대 경로
    private final String path;

    // 이벤트 발생 시각
    private final Instant eventTime;

    // ===== 이전(before) / 이후(after) 파일 상태 =====

    private final boolean existsBefore;
    private final boolean existsAfter;

    private final Long sizeBefore;
    private final Long sizeAfter;

    private final Double entropyBefore;
    private final Double entropyAfter;

    private final String extBefore;
    private final String extAfter;

    // ===== AI 분석 결과 (윈도우 단위 분석 결과를 이벤트에 태워서 저장) =====

    // AI 라벨 (SAFE / WARNING / DANGER / UNKNOWN 등)
    private final String aiLabel;

    // AI 위험도 점수 (0.0 ~ 1.0 등)
    private final Double aiScore;

    // AI 세부 설명/메시지
    private final String aiDetail;

    // ===== 편의 메서드 =====

    // 엔트로피 변화량 (after - before)
    public Double getEntropyDiff() {
        if (entropyBefore == null || entropyAfter == null) {
            return null;
        }
        return entropyAfter - entropyBefore;
    }

    // 크기 변화량 (after - before)
    public Long getSizeDiff() {
        if (sizeBefore == null || sizeAfter == null) {
            return null;
        }
        return sizeAfter - sizeBefore;
    }

    /**
     * 윈도우 단위로 계산된 AiResult 를 이 이벤트에 붙인 새로운 FileAnalysisResult 를 만든다.
     *
     * 원본은 그대로 두고, AI 필드만 채워진 새 인스턴스를 반환.
     */
    public FileAnalysisResult withAiResult(AiResult aiResult) {
        if (aiResult == null) {
            return this;
        }

        return FileAnalysisResult.builder()
                .ownerKey(this.ownerKey)
                .eventType(this.eventType)
                .path(this.path)
                .eventTime(this.eventTime)

                .existsBefore(this.existsBefore)
                .existsAfter(this.existsAfter)
                .sizeBefore(this.sizeBefore)
                .sizeAfter(this.sizeAfter)
                .entropyBefore(this.entropyBefore)
                .entropyAfter(this.entropyAfter)
                .extBefore(this.extBefore)
                .extAfter(this.extAfter)

                .aiLabel(aiResult.getLabel())
                .aiScore(aiResult.getScore())
                .aiDetail(aiResult.getDetail())
                .build();
    }
}
