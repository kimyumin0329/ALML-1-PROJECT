package com.watchserviceagent.watchservice_agent.support.dto;

import com.watchserviceagent.watchservice_agent.support.domain.FeedbackTicket;
import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

import java.time.ZoneId;
import java.time.format.DateTimeFormatter;

@Getter
@Builder
@ToString
public class FeedbackResponse {
    private final Long ticketId;
    private final String createdAt; // "yyyy-MM-dd HH:mm:ss"

    private static final DateTimeFormatter FMT =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")
                    .withZone(ZoneId.systemDefault());

    public static FeedbackResponse from(FeedbackTicket t) {
        return FeedbackResponse.builder()
                .ticketId(t.getId())
                .createdAt(t.getCreatedAt() == null ? "-" : FMT.format(t.getCreatedAt()))
                .build();
    }
}
