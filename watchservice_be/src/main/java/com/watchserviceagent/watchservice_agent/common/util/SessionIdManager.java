package com.watchserviceagent.watchservice_agent.common.util;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.util.UUID;

/**
 * 클래스 이름 : SessionIdManager
 * 기능 : 사용자(에이전트)별 고유 UUID(Session ID)를 생성하고 관리한다.
 *
 * - ./config/session_id.txt 파일에 UUID를 저장한다.
 * - 서버(에이전트)가 재시작되더라도 같은 파일을 읽어와서 동일한 UUID를 사용한다.
 * - 파일이 없거나 깨졌다면 새로 UUID를 생성하여 저장한다.
 *
 * 사용 예:
 *  - WatcherService 등에서 생성자 주입 후
 *      sessionIdManager.getSessionId()
 *    로 ownerKey(UUID)를 가져다 쓰면 된다.
 */
@Slf4j
@Component
public class SessionIdManager {

    /** 세션 ID를 저장할 디렉터리 (상대 경로 기준) */
    private static final Path CONFIG_DIR = Paths.get("./config");

    /** 세션 ID를 저장할 파일 경로 */
    private static final Path SESSION_FILE = CONFIG_DIR.resolve("session_id.txt");

    /**
     * 메모리에 유지할 세션 ID.
     * - 애플리케이션이 살아 있는 동안 변하지 않는다.
     * - 생성자에서 loadOrCreateSessionId()를 통해 초기화된다.
     */
    private final String sessionId;

    /**
     * 생성자
     * - 스프링이 빈을 만들 때 자동으로 호출된다.
     * - 이 시점에 session_id.txt를 읽거나 생성해서 sessionId 필드를 채운다.
     */
    public SessionIdManager() {
        this.sessionId = loadOrCreateSessionId();
        log.info("[SessionIdManager] Session ID 초기화 완료: {}", this.sessionId);
    }

    /**
     * session_id.txt 파일에서 Session ID를 읽어오거나, 없으면 새로 생성해서 파일에 저장한다.
     *
     * @return 메모리에 유지할 Session ID 문자열(UUID)
     */
    private String loadOrCreateSessionId() {
        try {
            // 1) config 디렉터리가 없으면 생성
            if (Files.notExists(CONFIG_DIR)) {
                Files.createDirectories(CONFIG_DIR);
                log.info("[SessionIdManager] config 디렉터리 생성: {}", CONFIG_DIR.toAbsolutePath());
            }

            // 2) session_id.txt 파일이 이미 존재하면: 읽어서 반환
            if (Files.exists(SESSION_FILE)) {
                String existingId = Files.readString(SESSION_FILE, StandardCharsets.UTF_8).trim();

                if (!existingId.isEmpty()) {
                    log.info("[SessionIdManager] 기존 Session ID 로드: {} (파일: {})",
                            existingId, SESSION_FILE.toAbsolutePath());
                    return existingId;
                } else {
                    log.warn("[SessionIdManager] session_id.txt 내용이 비어있음. 새 ID를 생성합니다.");
                }
            } else {
                log.info("[SessionIdManager] session_id.txt 파일이 없어 새 ID를 생성합니다. (파일: {})",
                        SESSION_FILE.toAbsolutePath());
            }

            // 3) 여기까지 왔다는 것은
            //    - 파일이 없거나, 내용이 비어있거나, 문제가 있는 경우이므로 새 UUID 생성
            String newId = UUID.randomUUID().toString();

            // 4) 생성한 UUID를 파일에 저장 (기존 내용은 덮어쓰기)
            Files.writeString(
                    SESSION_FILE,
                    newId,
                    StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.TRUNCATE_EXISTING,
                    StandardOpenOption.WRITE
            );

            log.info("[SessionIdManager] 새로운 Session ID 생성 및 저장: {} (파일: {})",
                    newId, SESSION_FILE.toAbsolutePath());

            return newId;

        } catch (IOException e) {
            // 파일 시스템 문제가 있어도 에이전트가 구동은 되게 해야 하므로
            // 임시 UUID를 반환하고, 로그로만 남긴다.
            log.error("[SessionIdManager] Session ID 로드/생성 중 오류 발생", e);
            String fallbackId = UUID.randomUUID().toString();
            log.warn("[SessionIdManager] 임시 Session ID를 사용합니다: {}", fallbackId);
            return fallbackId;
        }
    }

    /**
     * 현재 Session ID 조회.
     *
     * - 항상 동일한 값을 반환한다 (에이전트 프로세스가 살아 있는 동안).
     * - 외부에서는 이 값을 ownerKey로 사용하면 된다.
     */
    public String getSessionId() {
        return sessionId;
    }
}
