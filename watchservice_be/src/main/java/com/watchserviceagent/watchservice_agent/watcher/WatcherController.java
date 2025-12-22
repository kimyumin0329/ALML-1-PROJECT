package com.watchserviceagent.watchservice_agent.watcher;

import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

/**
 * 클래스 이름 : WatcherController
 * 기능 : 파일 감시 시작/중지 요청을 처리하는 REST API 엔드포인트를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
@Slf4j
@RestController
@RequestMapping("/watcher")
@RequiredArgsConstructor
public class WatcherController {

    private final WatcherService watcherService;

    /**
     * 함수 이름 : startWatchingPost
     * 기능 : POST 방식으로 파일 감시를 시작한다.
     * 매개변수 : folderPath - 감시할 폴더 경로
     * 반환값 : ResponseEntity<String> - 성공/실패 메시지
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostMapping("/start")
    public ResponseEntity<String> startWatchingPost(@RequestParam("folderPath") String folderPath) {
        return startInternal(folderPath, "POST");
    }

    /**
     * 함수 이름 : startWatchingGet
     * 기능 : GET 방식으로 파일 감시를 시작한다.
     * 매개변수 : folderPath - 감시할 폴더 경로
     * 반환값 : ResponseEntity<String> - 성공/실패 메시지
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/start")
    public ResponseEntity<String> startWatchingGet(@RequestParam("folderPath") String folderPath) {
        return startInternal(folderPath, "GET");
    }

    /**
     * 함수 이름 : startInternal
     * 기능 : 파일 감시 시작 요청을 내부적으로 처리한다.
     * 매개변수 : folderPath - 감시할 폴더 경로, method - HTTP 메서드 (로깅용)
     * 반환값 : ResponseEntity<String> - 성공/실패 메시지
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
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

    /**
     * 함수 이름 : stopWatchingPost
     * 기능 : POST 방식으로 파일 감시를 중지한다.
     * 매개변수 : 없음
     * 반환값 : ResponseEntity<String> - 성공/실패 메시지
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostMapping("/stop")
    public ResponseEntity<String> stopWatchingPost() {
        return stopInternal("POST");
    }

    /**
     * 함수 이름 : stopWatchingGet
     * 기능 : GET 방식으로 파일 감시를 중지한다.
     * 매개변수 : 없음
     * 반환값 : ResponseEntity<String> - 성공/실패 메시지
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/stop")
    public ResponseEntity<String> stopWatchingGet() {
        return stopInternal("GET");
    }

    /**
     * 함수 이름 : stopInternal
     * 기능 : 파일 감시 중지 요청을 내부적으로 처리한다.
     * 매개변수 : method - HTTP 메서드 (로깅용)
     * 반환값 : ResponseEntity<String> - 성공/실패 메시지
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
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
