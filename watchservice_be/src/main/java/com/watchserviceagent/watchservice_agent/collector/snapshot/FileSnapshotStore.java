// src/main/java/com/watchserviceagent/watchservice_agent/collector/snapshot/FileSnapshotStore.java
package com.watchserviceagent.watchservice_agent.collector.snapshot;

import lombok.Builder;
import lombok.Getter;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class FileSnapshotStore {

    @Getter
    @Builder
    public static class Snapshot {
        private final boolean exists;
        private final Long size;
        private final Double entropy;
        private final String ext;
        private final Long lastModifiedTime;
        private final String hash;
    }

    // ✅ “즉시 검사”로 만든 초기값(가능하면 유지)
    private final Map<String, Snapshot> baseline = new ConcurrentHashMap<>();
    // ✅ watcher/scan 이후 최신값(이벤트 들어오면 계속 갱신)
    private final Map<String, Snapshot> last = new ConcurrentHashMap<>();

    public Snapshot getBaseline(String path) {
        return baseline.get(path);
    }

    public Snapshot getLast(String path) {
        return last.get(path);
    }

    public void putBaselineIfAbsent(String path, Snapshot s) {
        if (path == null || s == null) return;
        baseline.putIfAbsent(path, s);
    }

    public void putLast(String path, Snapshot s) {
        if (path == null || s == null) return;
        last.put(path, s);
    }

    public void removeLast(String path) {
        if (path == null) return;
        last.remove(path);
    }

    /** CREATE 처음 본 파일은 baseline도 같이 세팅(초기정보가 없으면 의미가 없음) */
    public void ensureBaselineOnCreate(String path, Snapshot current) {
        putBaselineIfAbsent(path, current);
    }
}
