package com.watchserviceagent.watchservice_agent.ai.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

/**
 * AI 서버로 전송할 3초 윈도우 피처 벡터.
 *
 * JSON 전송 시에는 학습 CSV 헤더와 동일한 snake_case 키를 사용한다.
 */
@Getter
@Builder
@ToString
public class AiPayload {

    // 윈도우 동안 발생한 모든 MODIFY 이벤트 개수
    @JsonProperty("file_read_count")
    private int fileReadCount;

    // MODIFY 중 실제 내용이 바뀐 이벤트 개수
    @JsonProperty("file_write_count")
    private int fileWriteCount;

    @JsonProperty("file_delete_count")
    private int fileDeleteCount;

    @JsonProperty("file_rename_count")
    private int fileRenameCount;

    @JsonProperty("file_encrypt_like_count")
    private int fileEncryptLikeCount;

    @JsonProperty("changed_files_count")
    private int changedFilesCount;

    @JsonProperty("random_extension_flag")
    private int randomExtensionFlag;

    @JsonProperty("entropy_diff_mean")
    private double entropyDiffMean;

    @JsonProperty("file_size_diff_mean")
    private double fileSizeDiffMean;
}
