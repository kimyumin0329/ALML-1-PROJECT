package com.watchserviceagent.watchservice_agent.settings.dto;

import lombok.Getter;
import lombok.Setter;

/**
 * 예외(화이트리스트) 규칙 생성 요청 DTO.
 *
 * 프론트에서:
 *  POST /settings/exceptions
 *  {
 *    "type": "PATH",
 *    "pattern": "C:\\Users\\user\\SafeFolder",
 *    "memo": "백업 폴더"
 *  }
 */
@Getter
@Setter
public class ExceptionRuleRequest {

    private String type;    // PATH / EXT / ...
    private String pattern;
    private String memo;
}
