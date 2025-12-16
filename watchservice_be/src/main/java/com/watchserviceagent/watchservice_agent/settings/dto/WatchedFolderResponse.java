package com.watchserviceagent.watchservice_agent.settings.dto;

import com.watchserviceagent.watchservice_agent.settings.domain.WatchedFolder;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

/**
 * 감시 폴더 응답 DTO.
 *
 * - 프론트에서 리스트 렌더링에 사용.
 */
@Getter
@Builder
@ToString
public class WatchedFolderResponse {

    private final Long id;
    private final String name;
    private final String path;
    private final String createdAt; // "YYYY-MM-DD HH:mm:ss"

    private static final DateTimeFormatter DATE_TIME_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
                    .withZone(ZoneId.systemDefault());

    public static WatchedFolderResponse from(WatchedFolder folder) {
        String createdAtStr = folder.getCreatedAt() != null
                ? DATE_TIME_FORMATTER.format(folder.getCreatedAt())
                : "-";

        return WatchedFolderResponse.builder()
                .id(folder.getId())
                .name(folder.getName())
                .path(folder.getPath())
                .createdAt(createdAtStr)
                .build();
    }
}
