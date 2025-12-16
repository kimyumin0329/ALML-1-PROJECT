package com.watchserviceagent.watchservice_agent.support;

import com.watchserviceagent.watchservice_agent.support.domain.FeedbackTicket;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Repository;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.time.Instant;

@Repository
@RequiredArgsConstructor
@Slf4j
public class SupportRepository {

    private final JdbcTemplate jdbcTemplate;

    @PostConstruct
    public void init() {
        String sql = """
            CREATE TABLE IF NOT EXISTS feedback_ticket (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                owner_key  TEXT NOT NULL,
                name       TEXT,
                email      TEXT,
                content    TEXT NOT NULL,
                created_at INTEGER NOT NULL
            );
            """;
        jdbcTemplate.execute(sql);
        log.info("[SupportRepository] feedback_ticket 테이블 초기화 완료");
    }

    public FeedbackTicket insertFeedback(String ownerKey, String name, String email, String content) {
        long now = System.currentTimeMillis();

        String sql = """
            INSERT INTO feedback_ticket (owner_key, name, email, content, created_at)
            VALUES (?, ?, ?, ?, ?)
            """;
        jdbcTemplate.update(sql, ownerKey, blankToNull(name), blankToNull(email), content, now);

        Long id = jdbcTemplate.queryForObject("SELECT last_insert_rowid()", Long.class);

        return FeedbackTicket.builder()
                .id(id)
                .ownerKey(ownerKey)
                .name(blankToNull(name))
                .email(blankToNull(email))
                .content(content)
                .createdAt(Instant.ofEpochMilli(now))
                .build();
    }

    public FeedbackTicket findById(String ownerKey, Long id) {
        String sql = """
            SELECT id, owner_key, name, email, content, created_at
            FROM feedback_ticket
            WHERE owner_key = ? AND id = ?
            """;
        var list = jdbcTemplate.query(sql, new Object[]{ownerKey, id}, rowMapper());
        return list.isEmpty() ? null : list.get(0);
    }

    private RowMapper<FeedbackTicket> rowMapper() {
        return new RowMapper<>() {
            @Override
            public FeedbackTicket mapRow(ResultSet rs, int rowNum) throws SQLException {
                return FeedbackTicket.builder()
                        .id(rs.getLong("id"))
                        .ownerKey(rs.getString("owner_key"))
                        .name(rs.getString("name"))
                        .email(rs.getString("email"))
                        .content(rs.getString("content"))
                        .createdAt(Instant.ofEpochMilli(rs.getLong("created_at")))
                        .build();
            }
        };
    }

    private String blankToNull(String s) {
        if (s == null) return null;
        String t = s.trim();
        return t.isEmpty() ? null : t;
    }
}
