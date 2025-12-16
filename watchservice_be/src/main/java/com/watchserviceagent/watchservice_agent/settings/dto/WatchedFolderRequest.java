package com.watchserviceagent.watchservice_agent.settings.dto;

import lombok.Getter;
import lombok.Setter;

/**
 * 감시 폴더 생성 요청 DTO.
 *
 * 프론트에서:
 *  POST /settings/folders
 *  {
 *    "name": "문서 폴더",
 *    "path": "C:\\Users\\user\\Documents"
 *  }
 */
@Getter
@Setter
public class WatchedFolderRequest {

    private String name;
    private String path;
}
