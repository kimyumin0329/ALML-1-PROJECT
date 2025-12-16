package com.watchserviceagent.watchservice_agent.support;

import com.watchserviceagent.watchservice_agent.support.dto.FeedbackRequest;
import com.watchserviceagent.watchservice_agent.support.dto.FeedbackResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/support")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
public class SupportController {

    private final SupportService supportService;

    // POST /support/feedback
    @PostMapping("/feedback")
    public ResponseEntity<?> submit(@RequestBody FeedbackRequest req) {
        try {
            FeedbackResponse resp = supportService.submitFeedback(req);
            return ResponseEntity.ok(resp);
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().body(e.getMessage());
        } catch (Exception e) {
            log.error("[SupportController] feedback submit failed", e);
            return ResponseEntity.internalServerError().body("feedback submit failed");
        }
    }
}
