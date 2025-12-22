package com.watchserviceagent.watchservice_agent.collector.snapshot;

import lombok.Builder;
import lombok.Getter;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 클래스 이름 : FileSnapshotStore
 * 기능 : 파일의 이전 상태(baseline, last)를 메모리에 저장하여 변화를 추적할 수 있게 한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
public class FileSnapshotStore {

    /**
     * 클래스 이름 : Snapshot
     * 기능 : 파일의 스냅샷 정보를 담는 내부 클래스. 파일 존재 여부, 크기, 엔트로피, 확장자 등을 포함한다.
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
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

    private final Map<String, Snapshot> baseline = new ConcurrentHashMap<>();
    private final Map<String, Snapshot> last = new ConcurrentHashMap<>();

    /**
     * 함수 이름 : getBaseline
     * 기능 : 파일의 초기 상태(baseline) 스냅샷을 조회한다.
     * 매개변수 : path - 파일 경로
     * 반환값 : Snapshot - 초기 상태 스냅샷, 없으면 null
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public Snapshot getBaseline(String path) {
        return baseline.get(path);
    }

    /**
     * 함수 이름 : getLast
     * 기능 : 파일의 최신 상태(last) 스냅샷을 조회한다.
     * 매개변수 : path - 파일 경로
     * 반환값 : Snapshot - 최신 상태 스냅샷, 없으면 null
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public Snapshot getLast(String path) {
        return last.get(path);
    }

    /**
     * 함수 이름 : putBaselineIfAbsent
     * 기능 : baseline 스냅샷을 추가한다. 이미 존재하면 추가하지 않는다.
     * 매개변수 : path - 파일 경로, s - 스냅샷
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public void putBaselineIfAbsent(String path, Snapshot s) {
        if (path == null || s == null) return;
        baseline.putIfAbsent(path, s);
    }

    /**
     * 함수 이름 : putLast
     * 기능 : last 스냅샷을 업데이트한다.
     * 매개변수 : path - 파일 경로, s - 스냅샷
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public void putLast(String path, Snapshot s) {
        if (path == null || s == null) return;
        last.put(path, s);
    }

    /**
     * 함수 이름 : removeLast
     * 기능 : last 스냅샷을 제거한다.
     * 매개변수 : path - 파일 경로
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public void removeLast(String path) {
        if (path == null) return;
        last.remove(path);
    }

    /**
     * 함수 이름 : ensureBaselineOnCreate
     * 기능 : CREATE 이벤트로 처음 본 파일의 baseline을 설정한다.
     * 매개변수 : path - 파일 경로, current - 현재 스냅샷
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public void ensureBaselineOnCreate(String path, Snapshot current) {
        putBaselineIfAbsent(path, current);
    }
}
