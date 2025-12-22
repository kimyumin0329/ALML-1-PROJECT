package com.watchserviceagent.watchservice_agent.alerts;

import com.watchserviceagent.watchservice_agent.alerts.domain.Notification;
import com.watchserviceagent.watchservice_agent.alerts.dto.NotificationPageResponse;
import com.watchserviceagent.watchservice_agent.alerts.dto.NotificationResponse;
import com.watchserviceagent.watchservice_agent.common.util.SessionIdManager;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.List;
import java.util.Locale;
import java.util.NoSuchElementException;
import java.util.stream.Collectors;

/**
 * 클래스 이름 : NotificationService
 * 기능 : 알림(윈도우 단위 AI 분석 결과) 조회 및 통계를 처리하는 비즈니스 로직을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Service
@RequiredArgsConstructor
public class NotificationService {

    private final SessionIdManager sessionIdManager;
    private final NotificationRepository notificationRepository;

    private static final int MAX_PAGE_SIZE = 1000;
    private static final DateTimeFormatter DATE_TIME_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private static final DateTimeFormatter DATE_TIME_FORMATTER =
            DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss").withZone(ZoneId.systemDefault());

    /**
     * 함수 이름 : getNotifications
     * 기능 : 페이지네이션, 필터링, 정렬을 지원하는 알림 목록을 조회한다.
     * 매개변수 : page - 페이지 번호, size - 페이지 크기, from - 시작 날짜, to - 종료 날짜, level - 위험도 필터, keyword - 검색 키워드, sort - 정렬 기준
     * 반환값 : NotificationPageResponse - 페이지네이션된 알림 목록
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public NotificationPageResponse getNotifications(
            Integer page,
            Integer size,
            String from,
            String to,
            String level,
            String keyword,
            String sort
    ) {
        int p = (page == null || page < 1) ? 1 : page;
        int s = (size == null || size < 1) ? 50 : Math.min(size, MAX_PAGE_SIZE);

        SortInfo si = parseSort(sort);

        Long fromEpoch = parseFromToEpochStart(from);
        Long toEpoch = parseFromToEpochEnd(to);

        String ownerKey = sessionIdManager.getSessionId();

        String lv = normalizeLevel(level);

        long total = notificationRepository.countNotificationsByOwner(ownerKey, fromEpoch, toEpoch, keyword, lv);

        int offset = (p - 1) * s;
        List<Notification> notifications = notificationRepository.findNotificationsByOwner(
                ownerKey, fromEpoch, toEpoch, keyword, lv,
                si.field, si.dir, offset, s
        );

        List<NotificationResponse> items = notifications.stream()
                .map(this::toNotificationResponse)
                .collect(Collectors.toList());

        return NotificationPageResponse.builder()
                .items(items)
                .total(total)
                .page(p)
                .size(s)
                .build();
    }

    /**
     * 함수 이름 : getNotificationById
     * 기능 : ID로 단일 알림의 상세 정보를 조회한다.
     * 매개변수 : id - 알림 ID
     * 반환값 : NotificationResponse - 알림 상세 정보
     * 예외 : NoSuchElementException - 알림을 찾을 수 없을 때
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public NotificationResponse getNotificationById(long id) {
        String ownerKey = sessionIdManager.getSessionId();
        Notification notification = notificationRepository.findByIdAndOwner(ownerKey, id)
                .orElseThrow(() -> new NoSuchElementException("notification not found: id=" + id));
        return toNotificationResponse(notification);
    }

    /**
     * 함수 이름 : saveNotification
     * 기능 : 윈도우 단위 AI 분석 결과를 알림으로 저장한다.
     * 매개변수 : notification - 저장할 알림 엔티티
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public void saveNotification(Notification notification) {
        notificationRepository.insertNotification(notification);
    }

    private NotificationResponse toNotificationResponse(Notification notification) {
        return NotificationResponse.builder()
                .id(notification.getId())
                .windowStart(DATE_TIME_FORMATTER.format(notification.getWindowStart()))
                .windowEnd(DATE_TIME_FORMATTER.format(notification.getWindowEnd()))
                .createdAt(DATE_TIME_FORMATTER.format(notification.getCreatedAt()))
                .aiLabel(notification.getAiLabel())
                .aiScore(notification.getAiScore())
                .topFamily(notification.getTopFamily())
                .aiDetail(notification.getAiDetail())
                .affectedFilesCount(notification.getAffectedFilesCount())
                .affectedPaths(notification.getAffectedPaths())
                .build();
    }

    private String normalizeLevel(String level) {
        if (level == null) return null;
        String v = level.trim().toUpperCase(Locale.ROOT);
        if (v.isBlank() || v.equals("ALL")) return null;
        if (v.equals("DANGER") || v.equals("WARNING") || v.equals("SAFE")) return v;
        return null;
    }

    private SortInfo parseSort(String sort) {
        if (sort == null || sort.isBlank()) return new SortInfo("createdAt", "desc");
        String[] parts = sort.split(",");
        String field = parts.length > 0 ? parts[0].trim() : "createdAt";
        String dir = parts.length > 1 ? parts[1].trim() : "desc";
        if (field.isBlank()) field = "createdAt";
        if (dir.isBlank()) dir = "desc";
        return new SortInfo(field, dir);
    }

    private Long parseFromToEpochStart(String from) {
        if (from == null || from.isBlank()) return null;
        try {
            LocalDate d = LocalDate.parse(from.trim());
            return d.atStartOfDay(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (DateTimeParseException ignore) {}
        try {
            LocalDateTime dt = LocalDateTime.parse(from.trim(), DATE_TIME_FMT);
            return dt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (Exception ignore) {}
        return null;
    }

    private Long parseFromToEpochEnd(String to) {
        if (to == null || to.isBlank()) return null;
        try {
            LocalDate d = LocalDate.parse(to.trim());
            LocalDateTime end = d.atTime(23, 59, 59, 999_000_000);
            return end.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (DateTimeParseException ignore) {}
        try {
            LocalDateTime dt = LocalDateTime.parse(to.trim(), DATE_TIME_FMT);
            return dt.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (Exception ignore) {}
        return null;
    }

    private static class SortInfo {
        final String field;
        final String dir;
        SortInfo(String field, String dir) { this.field = field; this.dir = dir; }
    }
}

