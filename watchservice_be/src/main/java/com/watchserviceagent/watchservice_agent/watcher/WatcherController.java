package com.watchserviceagent.watchservice_agent.watcher;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

// 🔥 여기서도 한 번 더 CORS 허용
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
@Slf4j
@RestController
@RequestMapping("/watcher")
@RequiredArgsConstructor
public class WatcherController {

    private final WatcherService watcherService;

    // ================ 감시 시작 ================

    @PostMapping("/start")
    public ResponseEntity<String> startWatchingPost(@RequestParam("folderPath") String folderPath) {
        return startInternal(folderPath, "POST");
    }

    @GetMapping("/start")
    public ResponseEntity<String> startWatchingGet(@RequestParam("folderPath") String folderPath) {
        return startInternal(folderPath, "GET");
    }

    private ResponseEntity<String> startInternal(String folderPath, String method) {
        log.info("[WatcherController] 감시 시작 요청 (method={}) - folderPath={}", method, folderPath);
        try {
            watcherService.startWatching(folderPath);
            return ResponseEntity.ok("[Watcher] 감시를 시작했습니다: " + folderPath);
        } catch (Exception e) {
            log.error("[WatcherController] 감시 시작 실패", e);
            return ResponseEntity.internalServerError()
                    .body("[Watcher] 감시 시작 실패: " + e.getMessage());
        }
    }

    // ================ 감시 중지 ================

    @PostMapping("/stop")
    public ResponseEntity<String> stopWatchingPost() {
        return stopInternal("POST");
    }

    @GetMapping("/stop")
    public ResponseEntity<String> stopWatchingGet() {
        return stopInternal("GET");
    }

    private ResponseEntity<String> stopInternal(String method) {
        log.info("[WatcherController] 감시 중지 요청 (method={})", method);
        try {
            watcherService.stopWatching();
            return ResponseEntity.ok("[Watcher] 감시를 중지했습니다.");
        } catch (Exception e) {
            log.error("[WatcherController] 감시 중지 실패", e);
            return ResponseEntity.internalServerError()
                    .body("[Watcher] 감시 중지 실패: " + e.getMessage());
        }
    }
}
