package com.watchserviceagent.watchservice_agent.storage;

import com.watchserviceagent.watchservice_agent.storage.domain.Log;
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
 * 클래스 이름 : LogRepository
 * 기능 : SQLite 데이터베이스에 로그를 저장하고 조회하는 데이터 접근 계층을 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Repository
@RequiredArgsConstructor
@Slf4j
public class LogRepository {

    private final JdbcTemplate jdbcTemplate;

    /**
     * 함수 이름 : init
     * 기능 : 로그 테이블을 생성하고 필요한 컬럼을 추가한다. 애플리케이션 시작 시 자동 호출된다.
     * 매개변수 : 없음
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostConstruct
    public void init() {
        String sql = """
                CREATE TABLE IF NOT EXISTS log (
                    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_key          TEXT NOT NULL,
                    event_type         TEXT NOT NULL,
                    path               TEXT NOT NULL,
                    exists_flag        INTEGER NOT NULL,
                    size               INTEGER NOT NULL,
                    size_before        INTEGER,
                    size_after         INTEGER,
                    entropy_before     REAL,
                    entropy_after      REAL,
                    ext_before         TEXT,
                    ext_after          TEXT,
                    exists_before      INTEGER,
                    size_diff          INTEGER,
                    entropy_diff       REAL,
                    last_modified_time INTEGER NOT NULL,
                    hash               TEXT,
                    entropy            REAL,
                    ai_label           TEXT,
                    ai_score           REAL,
                    ai_detail          TEXT,
                    collected_at       INTEGER NOT NULL
                );
                """;
        jdbcTemplate.execute(sql);
        // 기존 테이블에 신규 컬럼 없을 수 있으므로 추가 시도(이미 있으면 무시)
        tryAddColumn("size_before", "INTEGER");
        tryAddColumn("size_after", "INTEGER");
        tryAddColumn("entropy_before", "REAL");
        tryAddColumn("entropy_after", "REAL");
        tryAddColumn("ext_before", "TEXT");
        tryAddColumn("ext_after", "TEXT");
        tryAddColumn("exists_before", "INTEGER");
        tryAddColumn("size_diff", "INTEGER");
        tryAddColumn("entropy_diff", "REAL");
        log.info("[LogRepository] log 테이블 초기화 완료");
    }

    /**
     * 함수 이름 : tryAddColumn
     * 기능 : 기존 테이블에 컬럼을 추가하려고 시도한다. 이미 존재하면 무시한다.
     * 매개변수 : column - 컬럼 이름, type - 컬럼 타입
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private void tryAddColumn(String column, String type) {
        try {
            jdbcTemplate.execute("ALTER TABLE log ADD COLUMN " + column + " " + type);
            log.info("[LogRepository] 컬럼 추가: {}", column);
        } catch (Exception ignore) {
            // 이미 존재하는 경우 등은 무시
        }
    }

    /**
     * 함수 이름 : insertLog
     * 기능 : 로그 엔티티를 데이터베이스에 삽입한다.
     * 매개변수 : logEntity - 저장할 로그 엔티티
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public void insertLog(Log logEntity) {
        String sql = """
                INSERT INTO log (
                    owner_key,
                    event_type,
                    path,
                    exists_flag,
                    size,
                    size_before,
                    size_after,
                    entropy_before,
                    entropy_after,
                    ext_before,
                    ext_after,
                    exists_before,
                    size_diff,
                    entropy_diff,
                    last_modified_time,
                    hash,
                    entropy,
                    ai_label,
                    ai_score,
                    ai_detail,
                    collected_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """;

        jdbcTemplate.update(
                sql,
                logEntity.getOwnerKey(),
                logEntity.getEventType(),
                logEntity.getPath(),
                logEntity.isExists() ? 1 : 0,
                logEntity.getSize(),
                logEntity.getSizeBefore(),
                logEntity.getSizeAfter(),
                logEntity.getEntropyBefore(),
                logEntity.getEntropyAfter(),
                logEntity.getExtBefore(),
                logEntity.getExtAfter(),
                logEntity.getExistsBefore() == null ? null : (logEntity.getExistsBefore() ? 1 : 0),
                logEntity.getSizeDiff(),
                logEntity.getEntropyDiff(),
                logEntity.getLastModifiedTime(),
                logEntity.getHash(),
                logEntity.getEntropy(),
                logEntity.getAiLabel(),
                logEntity.getAiScore(),
                logEntity.getAiDetail(),
                logEntity.getCollectedAt().toEpochMilli()
        );
    }

    public List<Log> findRecentLogsByOwner(String ownerKey, int limit) {
        String sql = """
                SELECT
                    id, owner_key, event_type, path, exists_flag, size,
                    size_before, size_after, entropy_before, entropy_after,
                    ext_before, ext_after, exists_before, size_diff, entropy_diff,
                    last_modified_time, hash, entropy, ai_label, ai_score, ai_detail, collected_at
                FROM log
                WHERE owner_key = ?
                ORDER BY collected_at DESC, id DESC
                LIMIT ?
                """;
        return jdbcTemplate.query(sql, new Object[]{ownerKey, limit}, logRowMapper());
    }

    public Optional<Log> findByIdAndOwner(String ownerKey, long id) {
        String sql = """
                SELECT
                    id, owner_key, event_type, path, exists_flag, size,
                    size_before, size_after, entropy_before, entropy_after,
                    ext_before, ext_after, exists_before, size_diff, entropy_diff,
                    last_modified_time, hash, entropy, ai_label, ai_score, ai_detail, collected_at
                FROM log
                WHERE owner_key = ? AND id = ?
                LIMIT 1
                """;
        List<Log> list = jdbcTemplate.query(sql, new Object[]{ownerKey, id}, logRowMapper());
        if (list.isEmpty()) return Optional.empty();
        return Optional.of(list.get(0));
    }

    public int deleteByIdAndOwner(String ownerKey, long id) {
        String sql = "DELETE FROM log WHERE owner_key = ? AND id = ?";
        return jdbcTemplate.update(sql, ownerKey, id);
    }

    public int deleteByIdsAndOwner(String ownerKey, List<Long> ids) {
        if (ids == null || ids.isEmpty()) return 0;

        StringBuilder sb = new StringBuilder("DELETE FROM log WHERE owner_key = ? AND id IN (");
        for (int i = 0; i < ids.size(); i++) {
            sb.append("?");
            if (i < ids.size() - 1) sb.append(",");
        }
        sb.append(")");

        List<Object> params = new ArrayList<>();
        params.add(ownerKey);
        params.addAll(ids);

        return jdbcTemplate.update(sb.toString(), params.toArray());
    }

    public List<Log> findByIdsAndOwner(String ownerKey, List<Long> ids, int limit) {
        if (ids == null || ids.isEmpty()) return List.of();

        List<Long> sliced = ids.size() > limit ? ids.subList(0, limit) : ids;

        StringBuilder sb = new StringBuilder("""
                SELECT
                    id, owner_key, event_type, path, exists_flag, size,
                    size_before, size_after, entropy_before, entropy_after,
                    ext_before, ext_after, exists_before, size_diff, entropy_diff,
                    last_modified_time, hash, entropy, ai_label, ai_score, ai_detail, collected_at
                FROM log
                WHERE owner_key = ? AND id IN (
                """);
        for (int i = 0; i < sliced.size(); i++) {
            sb.append("?");
            if (i < sliced.size() - 1) sb.append(",");
        }
        sb.append(") ORDER BY collected_at DESC, id DESC");

        List<Object> params = new ArrayList<>();
        params.add(ownerKey);
        params.addAll(sliced);

        return jdbcTemplate.query(sb.toString(), params.toArray(), logRowMapper());
    }

    public long countLogsByOwner(
            String ownerKey,
            Long fromEpochMs,
            Long toEpochMs,
            String keyword,
            String aiLabel,
            String eventType
    ) {
        SqlAndParams sp = buildWhere(ownerKey, fromEpochMs, toEpochMs, keyword, aiLabel, eventType);
        String sql = "SELECT COUNT(*) FROM log " + sp.whereClause;
        Long count = jdbcTemplate.queryForObject(sql, sp.params.toArray(), Long.class);
        return count == null ? 0 : count;
    }

    public List<Log> findLogsByOwner(
            String ownerKey,
            Long fromEpochMs,
            Long toEpochMs,
            String keyword,
            String aiLabel,
            String eventType,
            String sortField,
            String sortDir,
            int offset,
            int limit
    ) {
        SqlAndParams sp = buildWhere(ownerKey, fromEpochMs, toEpochMs, keyword, aiLabel, eventType);
        String orderBy = buildOrderBy(sortField, sortDir);

        String sql = """
                SELECT
                    id, owner_key, event_type, path, exists_flag, size,
                    size_before, size_after, entropy_before, entropy_after,
                    ext_before, ext_after, exists_before, size_diff, entropy_diff,
                    last_modified_time, hash, entropy, ai_label, ai_score, ai_detail, collected_at
                FROM log
                """ + sp.whereClause + " " + orderBy + " LIMIT ? OFFSET ?";

        List<Object> params = new ArrayList<>(sp.params);
        params.add(limit);
        params.add(offset);

        return jdbcTemplate.query(sql, params.toArray(), logRowMapper());
    }

    private String buildOrderBy(String sortField, String sortDir) {
        String col;
        if (sortField == null) sortField = "collectedAt";
        if (sortDir == null) sortDir = "desc";

        switch (sortField) {
            case "aiScore" -> col = "ai_score";
            case "entropy" -> col = "entropy";
            case "size" -> col = "size";
            case "id" -> col = "id";
            case "collectedAt" -> col = "collected_at";
            default -> col = "collected_at";
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
            String aiLabel,
            String eventType
    ) {
        StringBuilder where = new StringBuilder("WHERE owner_key = ?");
        List<Object> params = new ArrayList<>();
        params.add(ownerKey);

        if (aiLabel != null && !aiLabel.isBlank()) {
            where.append(" AND ai_label = ?");
            params.add(aiLabel.trim());
        }
        if (eventType != null && !eventType.isBlank()) {
            where.append(" AND event_type = ?");
            params.add(eventType.trim());
        }

        if (fromEpochMs != null) {
            where.append(" AND collected_at >= ?");
            params.add(fromEpochMs);
        }
        if (toEpochMs != null) {
            where.append(" AND collected_at <= ?");
            params.add(toEpochMs);
        }

        if (keyword != null && !keyword.isBlank()) {
            String like = "%" + keyword + "%";
            where.append(" AND (path LIKE ? OR event_type LIKE ? OR ai_detail LIKE ?)");
            params.add(like);
            params.add(like);
            params.add(like);
        }

        return new SqlAndParams(" " + where + " ", params);
    }

    private RowMapper<Log> logRowMapper() {
        return new RowMapper<>() {
            @Override
            public Log mapRow(ResultSet rs, int rowNum) throws SQLException {
                // SQLite는 INTEGER/REAL 컬럼을 Integer, Long, Double 등으로 돌려줄 수 있으므로
                // Number 기반으로 안전하게 캐스팅한다.
                Number sizeBeforeNum = (Number) rs.getObject("size_before");
                Number sizeAfterNum = (Number) rs.getObject("size_after");
                Number sizeDiffNum = (Number) rs.getObject("size_diff");
                Number entropyBeforeNum = (Number) rs.getObject("entropy_before");
                Number entropyAfterNum = (Number) rs.getObject("entropy_after");
                Number entropyDiffNum = (Number) rs.getObject("entropy_diff");

                Long sizeBefore = sizeBeforeNum == null ? null : sizeBeforeNum.longValue();
                Long sizeAfter = sizeAfterNum == null ? null : sizeAfterNum.longValue();
                Long sizeDiff = sizeDiffNum == null ? null : sizeDiffNum.longValue();
                Double entropyBefore = entropyBeforeNum == null ? null : entropyBeforeNum.doubleValue();
                Double entropyAfter = entropyAfterNum == null ? null : entropyAfterNum.doubleValue();
                Double entropyDiff = entropyDiffNum == null ? null : entropyDiffNum.doubleValue();

                String extBefore = rs.getString("ext_before");
                String extAfter = rs.getString("ext_after");
                Boolean existsBefore = rs.getObject("exists_before") == null
                        ? null
                        : rs.getInt("exists_before") != 0;

                return Log.builder()
                        .id(rs.getLong("id"))
                        .ownerKey(rs.getString("owner_key"))
                        .eventType(rs.getString("event_type"))
                        .path(rs.getString("path"))
                        .exists(rs.getInt("exists_flag") != 0)
                        .size(rs.getLong("size"))
                        .sizeBefore(sizeBefore)
                        .sizeAfter(sizeAfter)
                        .entropyBefore(entropyBefore)
                        .entropyAfter(entropyAfter)
                        .extBefore(extBefore)
                        .extAfter(extAfter)
                        .existsBefore(existsBefore)
                        .sizeDiff(sizeDiff)
                        .entropyDiff(entropyDiff)
                        .lastModifiedTime(rs.getLong("last_modified_time"))
                        .hash(rs.getString("hash"))
                        .entropy(rs.getObject("entropy") != null ? rs.getDouble("entropy") : null)
                        .aiLabel(rs.getString("ai_label"))
                        .aiScore(rs.getObject("ai_score") != null ? rs.getDouble("ai_score") : null)
                        .aiDetail(rs.getString("ai_detail"))
                        .collectedAt(Instant.ofEpochMilli(rs.getLong("collected_at")))
                        .build();
            }
        };
    }

    private static class SqlAndParams {
        final String whereClause;
        final List<Object> params;

        SqlAndParams(String whereClause, List<Object> params) {
            this.whereClause = whereClause;
            this.params = params;
        }
    }

    // ====== ALERTS 전용 API 지원 메서드들 ======

    /**
     * 알림(Alerts) = ai_label이 있는 로그
     * level: DANGER | WARNING | SAFE | null(전체)
     */
    public long countAlertsByOwner(
            String ownerKey,
            Long fromEpochMs,
            Long toEpochMs,
            String keyword,
            String level
    ) {
        SqlAndParams sp = buildAlertWhere(ownerKey, fromEpochMs, toEpochMs, keyword, level);
        String sql = "SELECT COUNT(*) FROM log " + sp.whereClause;
        Long count = jdbcTemplate.queryForObject(sql, sp.params.toArray(), Long.class);
        return count == null ? 0 : count;
    }

    /**
     * 알림 목록 조회 (페이지/정렬)
     */
    public List<Log> findAlertsByOwner(
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
        SqlAndParams sp = buildAlertWhere(ownerKey, fromEpochMs, toEpochMs, keyword, level);
        String orderBy = buildOrderBy(sortField, sortDir);

        String sql = """
            SELECT
                id, owner_key, event_type, path, exists_flag, size,
                size_before, size_after, entropy_before, entropy_after,
                ext_before, ext_after, exists_before, size_diff, entropy_diff,
                last_modified_time, hash, entropy, ai_label, ai_score, ai_detail, collected_at
            FROM log
            """ + sp.whereClause + " " + orderBy + " LIMIT ? OFFSET ?";

        List<Object> params = new ArrayList<>(sp.params);
        params.add(limit);
        params.add(offset);

        return jdbcTemplate.query(sql, params.toArray(), logRowMapper());
    }

    /**
     * 알림 상세(단건) 조회 = ai_label 있는 로그만 허용
     */
    public Optional<Log> findAlertByIdAndOwner(String ownerKey, long id) {
        String sql = """
            SELECT
                id, owner_key, event_type, path, exists_flag, size,
                size_before, size_after, entropy_before, entropy_after,
                ext_before, ext_after, exists_before, size_diff, entropy_diff,
                last_modified_time, hash, entropy, ai_label, ai_score, ai_detail, collected_at
            FROM log
            WHERE owner_key = ?
              AND id = ?
              AND ai_label IS NOT NULL
              AND TRIM(ai_label) <> ''
            LIMIT 1
            """;
        List<Log> list = jdbcTemplate.query(sql, new Object[]{ownerKey, id}, logRowMapper());
        if (list.isEmpty()) return Optional.empty();
        return Optional.of(list.get(0));
    }

    /**
     * 알림 통계(daily/weekly)
     * - WARNING / DANGER 카운트만 반환 (series)
     */
    public List<AlertStatRow> getAlertStats(String ownerKey, Long fromEpochMs, Long toEpochMs, String range) {
        String rg = (range == null) ? "daily" : range.trim().toLowerCase(Locale.ROOT);

        // SQLite: collected_at(ms) -> seconds 로 변환해서 버킷 생성 (localtime 적용)
        String bucketExpr;
        if ("weekly".equals(rg)) {
            bucketExpr = "strftime('%Y-W%W', collected_at/1000, 'unixepoch', 'localtime')";
        } else {
            bucketExpr = "strftime('%Y-%m-%d', collected_at/1000, 'unixepoch', 'localtime')";
            rg = "daily";
        }

        StringBuilder where = new StringBuilder("""
            WHERE owner_key = ?
              AND ai_label IS NOT NULL
              AND TRIM(ai_label) <> ''
            """);

        List<Object> params = new ArrayList<>();
        params.add(ownerKey);

        if (fromEpochMs != null) {
            where.append(" AND collected_at >= ?");
            params.add(fromEpochMs);
        }
        if (toEpochMs != null) {
            where.append(" AND collected_at <= ?");
            params.add(toEpochMs);
        }

        String sql = """
            SELECT
              %s AS bucket,
              SUM(CASE WHEN ai_label = 'WARNING' THEN 1 ELSE 0 END) AS warning,
              SUM(CASE WHEN ai_label = 'DANGER' THEN 1 ELSE 0 END) AS danger
            FROM log
            %s
            GROUP BY bucket
            ORDER BY bucket ASC
            """.formatted(bucketExpr, where);

        return jdbcTemplate.query(sql, params.toArray(), (rs, rowNum) ->
                new AlertStatRow(
                        rs.getString("bucket"),
                        rs.getLong("warning"),
                        rs.getLong("danger")
                )
        );
    }

    /** Alert stats row shape for AlertService */
    public record AlertStatRow(String bucket, long warning, long danger) {}

    /**
     * 알림 전용 WHERE builder
     * - ai_label이 있는 로그만
     * - level(DANGER/WARNING/SAFE)이 있으면 ai_label=level
     */
    private SqlAndParams buildAlertWhere(
            String ownerKey,
            Long fromEpochMs,
            Long toEpochMs,
            String keyword,
            String level
    ) {
        StringBuilder where = new StringBuilder("""
            WHERE owner_key = ?
              AND ai_label IS NOT NULL
              AND TRIM(ai_label) <> ''
            """);

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
            where.append(" AND collected_at >= ?");
            params.add(fromEpochMs);
        }
        if (toEpochMs != null) {
            where.append(" AND collected_at <= ?");
            params.add(toEpochMs);
        }

        if (keyword != null && !keyword.isBlank()) {
            String like = "%" + keyword + "%";
            where.append(" AND (path LIKE ? OR event_type LIKE ? OR ai_detail LIKE ?)");
            params.add(like);
            params.add(like);
            params.add(like);
        }

        return new SqlAndParams(" " + where + " ", params);
    }

}
