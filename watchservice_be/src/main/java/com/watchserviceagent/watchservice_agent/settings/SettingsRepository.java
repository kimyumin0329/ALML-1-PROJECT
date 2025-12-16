package com.watchserviceagent.watchservice_agent.settings;

import com.watchserviceagent.watchservice_agent.settings.domain.WatchedFolder;
import com.watchserviceagent.watchservice_agent.settings.domain.ExceptionRule;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Instant;
import java.util.List;

/**
 * 설정(감시 폴더, 예외 규칙) 관련 SQLite 접근 레이어.
 *
 * - watched_folder
 * - exception_rule
 */
@Repository
@RequiredArgsConstructor
@Slf4j
public class SettingsRepository {

    private final JdbcTemplate jdbcTemplate;

    @PostConstruct
    public void init() {
        createWatchedFolderTable();
        createExceptionRuleTable();
    }

    private void createWatchedFolderTable() {
        String sql = """
                CREATE TABLE IF NOT EXISTS watched_folder (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_key  TEXT NOT NULL,
                    name       TEXT NOT NULL,
                    path       TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                );
                """;
        jdbcTemplate.execute(sql);
        log.info("[SettingsRepository] watched_folder 테이블 초기화 완료");
    }

    private void createExceptionRuleTable() {
        String sql = """
                CREATE TABLE IF NOT EXISTS exception_rule (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_key  TEXT NOT NULL,
                    type       TEXT NOT NULL,
                    pattern    TEXT NOT NULL,
                    memo       TEXT,
                    created_at INTEGER NOT NULL
                );
                """;
        jdbcTemplate.execute(sql);
        log.info("[SettingsRepository] exception_rule 테이블 초기화 완료");
    }

    // ===== 감시 폴더 =====

    public List<WatchedFolder> findWatchedFolders(String ownerKey) {
        String sql = """
                SELECT id, owner_key, name, path, created_at
                FROM watched_folder
                WHERE owner_key = ?
                ORDER BY id ASC
                """;
        return jdbcTemplate.query(sql, new Object[]{ownerKey}, watchedFolderRowMapper());
    }

    public WatchedFolder insertWatchedFolder(String ownerKey, String name, String path) {
        long now = System.currentTimeMillis();

        String sql = """
                INSERT INTO watched_folder (
                    owner_key, name, path, created_at
                ) VALUES (?, ?, ?, ?)
                """;
        jdbcTemplate.update(sql, ownerKey, name, path, now);

        // SQLite 는 last_insert_rowid() 로 마지막 PK 조회 가능
        Long id = jdbcTemplate.queryForObject("SELECT last_insert_rowid()", Long.class);
        return WatchedFolder.builder()
                .id(id)
                .ownerKey(ownerKey)
                .name(name)
                .path(path)
                .createdAt(Instant.ofEpochMilli(now))
                .build();
    }

    public void deleteWatchedFolder(String ownerKey, Long id) {
        String sql = "DELETE FROM watched_folder WHERE owner_key = ? AND id = ?";
        jdbcTemplate.update(sql, ownerKey, id);
    }

    private RowMapper<WatchedFolder> watchedFolderRowMapper() {
        return new RowMapper<>() {
            @Override
            public WatchedFolder mapRow(ResultSet rs, int rowNum) throws SQLException {
                return WatchedFolder.builder()
                        .id(rs.getLong("id"))
                        .ownerKey(rs.getString("owner_key"))
                        .name(rs.getString("name"))
                        .path(rs.getString("path"))
                        .createdAt(Instant.ofEpochMilli(rs.getLong("created_at")))
                        .build();
            }
        };
    }

    // ===== 예외 규칙 =====

    public List<ExceptionRule> findExceptionRules(String ownerKey) {
        String sql = """
                SELECT id, owner_key, type, pattern, memo, created_at
                FROM exception_rule
                WHERE owner_key = ?
                ORDER BY id ASC
                """;
        return jdbcTemplate.query(sql, new Object[]{ownerKey}, exceptionRuleRowMapper());
    }

    public ExceptionRule insertExceptionRule(String ownerKey, String type, String pattern, String memo) {
        long now = System.currentTimeMillis();

        String sql = """
                INSERT INTO exception_rule (
                    owner_key, type, pattern, memo, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """;
        jdbcTemplate.update(sql, ownerKey, type, pattern, memo, now);

        Long id = jdbcTemplate.queryForObject("SELECT last_insert_rowid()", Long.class);
        return ExceptionRule.builder()
                .id(id)
                .ownerKey(ownerKey)
                .type(type)
                .pattern(pattern)
                .memo(memo)
                .createdAt(Instant.ofEpochMilli(now))
                .build();
    }

    public void deleteExceptionRule(String ownerKey, Long id) {
        String sql = "DELETE FROM exception_rule WHERE owner_key = ? AND id = ?";
        jdbcTemplate.update(sql, ownerKey, id);
    }

    private RowMapper<ExceptionRule> exceptionRuleRowMapper() {
        return new RowMapper<>() {
            @Override
            public ExceptionRule mapRow(ResultSet rs, int rowNum) throws SQLException {
                return ExceptionRule.builder()
                        .id(rs.getLong("id"))
                        .ownerKey(rs.getString("owner_key"))
                        .type(rs.getString("type"))
                        .pattern(rs.getString("pattern"))
                        .memo(rs.getString("memo"))
                        .createdAt(Instant.ofEpochMilli(rs.getLong("created_at")))
                        .build();
            }
        };
    }
}
