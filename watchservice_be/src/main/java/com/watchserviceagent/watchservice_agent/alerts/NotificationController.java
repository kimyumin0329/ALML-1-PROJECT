package com.watchserviceagent.watchservice_agent.alerts;

import com.watchserviceagent.watchservice_agent.alerts.dto.NotificationPageResponse;
import com.watchserviceagent.watchservice_agent.alerts.dto.NotificationResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

/**
 * 클래스 이름 : NotificationController
 * 기능 : 알림(윈도우 단위 AI 분석 결과) 조회를 제공하는 REST API 엔드포인트를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
@RestController
@RequestMapping("/notifications")
@RequiredArgsConstructor
public class NotificationController {

    private final NotificationService notificationService;

    /**
     * 함수 이름 : getNotifications
     * 기능 : 페이지네이션, 필터링, 정렬을 지원하는 알림 목록을 조회한다.
     * 매개변수 : page - 페이지 번호 (1-based), size - 페이지 크기, from - 시작 날짜, to - 종료 날짜, level - 위험도 필터 (ALL|DANGER|WARNING|SAFE), keyword - 검색 키워드, sort - 정렬 기준
     * 반환값 : NotificationPageResponse - 페이지네이션된 알림 목록
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping
    public NotificationPageResponse getNotifications(
            @RequestParam(name = "page", required = false) Integer page,
            @RequestParam(name = "size", required = false) Integer size,
            @RequestParam(name = "from", required = false) String from,
            @RequestParam(name = "to", required = false) String to,
            @RequestParam(name = "level", required = false) String level,
            @RequestParam(name = "keyword", required = false) String keyword,
            @RequestParam(name = "sort", required = false) String sort
    ) {
        return notificationService.getNotifications(page, size, from, to, level, keyword, sort);
    }

    /**
     * 함수 이름 : getNotification
     * 기능 : ID로 단일 알림의 상세 정보를 조회한다.
     * 매개변수 : id - 알림 ID
     * 반환값 : NotificationResponse - 알림 상세 정보
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/{id}")
    public NotificationResponse getNotification(@PathVariable("id") long id) {
        return notificationService.getNotificationById(id);
    }
}

