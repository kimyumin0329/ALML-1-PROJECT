package com.watchserviceagent.watchservice_agent.alerts;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.watchserviceagent.watchservice_agent.alerts.domain.Notification;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Instant;
import java.util.*;

/**
 * 클래스 이름 : NotificationRepository
 * 기능 : SQLite 데이터베이스에 알림(Notification)을 저장하고 조회하는 데이터 접근 계층을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Repository
@RequiredArgsConstructor
@Slf4j
public class NotificationRepository {

    private final JdbcTemplate jdbcTemplate;
    private final ObjectMapper objectMapper = new ObjectMapper();

    /**
     * 함수 이름 : init
     * 기능 : 알림 테이블을 생성한다. 애플리케이션 시작 시 자동 호출된다.
     * 매개변수 : 없음
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostConstruct
    public void init() {
        String sql = """
                CREATE TABLE IF NOT EXISTS notification (
                    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_key             TEXT NOT NULL,
                    window_start          INTEGER NOT NULL,
                    window_end            INTEGER NOT NULL,
                    created_at            INTEGER NOT NULL,
                    ai_label              TEXT,
                    ai_score              REAL,
                    top_family            TEXT,
                    ai_detail             TEXT,
                    affected_files_count  INTEGER NOT NULL,
                    affected_paths        TEXT NOT NULL
                );
                """;
        jdbcTemplate.execute(sql);
        log.info("[NotificationRepository] notification 테이블 초기화 완료");
    }

    /**
     * 함수 이름 : insertNotification
     * 기능 : 알림 엔티티를 데이터베이스에 삽입한다.
     * 매개변수 : notification - 저장할 알림 엔티티
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public void insertNotification(Notification notification) {
        String sql = """
                INSERT INTO notification (
                    owner_key,
                    window_start,
                    window_end,
                    created_at,
                    ai_label,
                    ai_score,
                    top_family,
                    ai_detail,
                    affected_files_count,
                    affected_paths
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """;

        try {
            String affectedPathsJson = objectMapper.writeValueAsString(notification.getAffectedPaths());

            jdbcTemplate.update(
                    sql,
                    notification.getOwnerKey(),
                    notification.getWindowStart().toEpochMilli(),
                    notification.getWindowEnd().toEpochMilli(),
                    notification.getCreatedAt().toEpochMilli(),
                    notification.getAiLabel(),
                    notification.getAiScore(),
                    notification.getTopFamily(),
                    notification.getAiDetail(),
                    notification.getAffectedFilesCount(),
                    affectedPathsJson
            );
        } catch (Exception e) {
            log.error("[NotificationRepository] insertNotification 실패", e);
            throw new RuntimeException("알림 저장 실패", e);
        }
    }

    /**
     * 함수 이름 : findByIdAndOwner
     * 기능 : ID와 ownerKey로 단일 알림을 조회한다.
     * 매개변수 : ownerKey - 소유자 키, id - 알림 ID
     * 반환값 : Optional<Notification> - 조회된 알림 (없으면 empty)
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public Optional<Notification> findByIdAndOwner(String ownerKey, long id) {
        String sql = """
                SELECT
                    id, owner_key, window_start, window_end, created_at,
                    ai_label, ai_score, top_family, ai_detail,
                    affected_files_count, affected_paths
                FROM notification
                WHERE owner_key = ? AND id = ?
                LIMIT 1
                """;
        List<Notification> list = jdbcTemplate.query(sql, new Object[]{ownerKey, id}, notificationRowMapper());
        if (list.isEmpty()) return Optional.empty();
        return Optional.of(list.get(0));
    }

    /**
     * 함수 이름 : countNotificationsByOwner
     * 기능 : 필터 조건에 맞는 알림 개수를 조회한다.
     * 매개변수 : ownerKey - 소유자 키, fromEpochMs - 시작 시간(epoch ms), toEpochMs - 종료 시간(epoch ms), keyword - 검색 키워드, level - 위험도 필터
     * 반환값 : long - 알림 개수
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public long countNotificationsByOwner(
            String ownerKey,
            Long fromEpochMs,
            Long toEpochMs,
            String keyword,
            String level
    ) {
        SqlAndParams sp = buildWhere(ownerKey, fromEpochMs, toEpochMs, keyword, level);
        String sql = "SELECT COUNT(*) FROM notification " + sp.whereClause;
        Long count = jdbcTemplate.queryForObject(sql, sp.params.toArray(), Long.class);
        return count == null ? 0 : count;
    }

    /**
     * 함수 이름 : findNotificationsByOwner
     * 기능 : 필터 조건에 맞는 알림 목록을 페이지네이션하여 조회한다.
     * 매개변수 : ownerKey - 소유자 키, fromEpochMs - 시작 시간, toEpochMs - 종료 시간, keyword - 검색 키워드, level - 위험도 필터, sortField - 정렬 필드, sortDir - 정렬 방향, offset - 오프셋, limit - 제한 개수
     * 반환값 : List<Notification> - 알림 목록
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public List<Notification> findNotificationsByOwner(
            String ownerKey,
            Long fromEpochMs,
            Long toEpochMs,
            String keyword,
            String level,
            String sortField,
            String sortDir,
            int offset,
            int limit
    ) {
        SqlAndParams sp = buildWhere(ownerKey, fromEpochMs, toEpochMs, keyword, level);
        String orderBy = buildOrderBy(sortField, sortDir);

        String sql = """
                SELECT
                    id, owner_key, window_start, window_end, created_at,
                    ai_label, ai_score, top_family, ai_detail,
                    affected_files_count, affected_paths
                FROM notification
                """ + sp.whereClause + " " + orderBy + " LIMIT ? OFFSET ?";

        List<Object> params = new ArrayList<>(sp.params);
        params.add(limit);
        params.add(offset);

        return jdbcTemplate.query(sql, params.toArray(), notificationRowMapper());
    }

    private RowMapper<Notification> notificationRowMapper() {
        return new RowMapper<>() {
            @Override
            public Notification mapRow(ResultSet rs, int rowNum) throws SQLException {
                String affectedPathsJson = rs.getString("affected_paths");
                List<String> affectedPaths = Collections.emptyList();
                try {
                    if (affectedPathsJson != null && !affectedPathsJson.isBlank()) {
                        affectedPaths = objectMapper.readValue(affectedPathsJson, new TypeReference<List<String>>() {});
                    }
                } catch (Exception e) {
                    log.warn("[NotificationRepository] affectedPaths JSON 파싱 실패: {}", affectedPathsJson, e);
                }

                return Notification.builder()
                        .id(rs.getLong("id"))
                        .ownerKey(rs.getString("owner_key"))
                        .windowStart(Instant.ofEpochMilli(rs.getLong("window_start")))
                        .windowEnd(Instant.ofEpochMilli(rs.getLong("window_end")))
                        .createdAt(Instant.ofEpochMilli(rs.getLong("created_at")))
                        .aiLabel(rs.getString("ai_label"))
                        .aiScore(rs.getObject("ai_score") != null ? rs.getDouble("ai_score") : null)
                        .topFamily(rs.getString("top_family"))
                        .aiDetail(rs.getString("ai_detail"))
                        .affectedFilesCount(rs.getInt("affected_files_count"))
                        .affectedPaths(affectedPaths)
                        .build();
            }
        };
    }

    private String buildOrderBy(String sortField, String sortDir) {
        String col;
        if (sortField == null) sortField = "createdAt";
        if (sortDir == null) sortDir = "desc";

        switch (sortField) {
            case "aiScore" -> col = "ai_score";
            case "windowStart" -> col = "window_start";
            case "windowEnd" -> col = "window_end";
            case "createdAt" -> col = "created_at";
            case "id" -> col = "id";
            default -> col = "created_at";
        }

        String dir = "DESC";
        if ("asc".equalsIgnoreCase(sortDir) || "ASC".equalsIgnoreCase(sortDir)) dir = "ASC";

        return "ORDER BY " + col + " " + dir + ", id DESC";
    }

    private SqlAndParams buildWhere(
            String ownerKey,
            Long fromEpochMs,
            Long toEpochMs,
            String keyword,
            String level
    ) {
        StringBuilder where = new StringBuilder("WHERE owner_key = ?");
        List<Object> params = new ArrayList<>();
        params.add(ownerKey);

        if (level != null && !level.isBlank()) {
            String lv = level.trim().toUpperCase(Locale.ROOT);
            if (!lv.equals("ALL") && (lv.equals("DANGER") || lv.equals("WARNING") || lv.equals("SAFE"))) {
                where.append(" AND ai_label = ?");
                params.add(lv);
            }
        }

        if (fromEpochMs != null) {
            where.append(" AND created_at >= ?");
            params.add(fromEpochMs);
        }
        if (toEpochMs != null) {
            where.append(" AND created_at <= ?");
            params.add(toEpochMs);
        }

        if (keyword != null && !keyword.isBlank()) {
            String like = "%" + keyword + "%";
            where.append(" AND (affected_paths LIKE ? OR ai_detail LIKE ? OR top_family LIKE ?)");
            params.add(like);
            params.add(like);
            params.add(like);
        }

        return new SqlAndParams(" " + where + " ", params);
    }

    private static class SqlAndParams {
        final String whereClause;
        final List<Object> params;

        SqlAndParams(String whereClause, List<Object> params) {
            this.whereClause = whereClause;
            this.params = params;
        }
    }
}

