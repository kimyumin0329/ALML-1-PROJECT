package com.watchserviceagent.watchservice_agent.support;

import com.watchserviceagent.watchservice_agent.common.util.SessionIdManager;
import com.watchserviceagent.watchservice_agent.support.domain.FeedbackTicket;
import com.watchserviceagent.watchservice_agent.support.dto.FeedbackRequest;
import com.watchserviceagent.watchservice_agent.support.dto.FeedbackResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class SupportService {

    private final SessionIdManager sessionIdManager;
    private final SupportRepository supportRepository;

    public FeedbackResponse submitFeedback(FeedbackRequest req) {
        String ownerKey = sessionIdManager.getSessionId();

        String content = req.getContent() == null ? "" : req.getContent().trim();
        if (content.isEmpty()) {
            throw new IllegalArgumentException("content is required");
        }

        FeedbackTicket ticket = supportRepository.insertFeedback(
                ownerKey,
                req.getName(),
                req.getEmail(),
                content
        );

        log.info("[SupportService] feedback submitted ticketId={}", ticket.getId());
        return FeedbackResponse.from(ticket);
    }
}
