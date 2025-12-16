package com.watchserviceagent.watchservice_agent.support.dto;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class FeedbackRequest {
    private String name;    // optional
    private String email;   // optional
    private String content; // required
}
