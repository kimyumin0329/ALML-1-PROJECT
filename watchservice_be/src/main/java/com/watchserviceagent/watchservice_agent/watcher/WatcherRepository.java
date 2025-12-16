package com.watchserviceagent.watchservice_agent.watcher;

import org.springframework.stereotype.Repository;

/**
 * 감시 관련 설정/이력을 저장/조회하는 Repository.
 *
 * 현재 구현:
 *  - 예제/졸업프로젝트 단계에서는 "마지막 감시 경로"만 메모리에 보관.
 *  - 실제 서비스에서는 SQLite 테이블이나 설정 파일로 persistence 할 수 있다.
 */
@Repository
public class WatcherRepository {

    /**
     * 마지막으로 감시를 시작한 경로를 단순히 메모리에 저장.
     * (서버 재시작 시 사라진다는 점에 유의)
     */
    private String lastWatchedPath;

    /**
     * 감시 대상 경로 저장.
     *
     * @param folderPath 사용자가 감시를 시작한 루트 경로
     */
    public synchronized void savePath(String folderPath) {
        this.lastWatchedPath = folderPath;
    }

    /**
     * 마지막으로 감시한 루트 경로 조회.
     *
     * @return 마지막 감시 루트 경로, 없으면 null
     */
    public synchronized String getLastWatchedPath() {
        return lastWatchedPath;
    }
}
