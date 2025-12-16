package com.watchserviceagent.watchservice_agent.support.domain;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.Instant;

@Getter
@Builder
@ToString
public class FeedbackTicket {
    private final Long id;
    private final String ownerKey;

    private final String name;
    private final String email;
    private final String content;

    private final Instant createdAt;
}
