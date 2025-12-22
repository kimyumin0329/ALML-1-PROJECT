package com.watchserviceagent.watchservice_agent.watcher;

import com.watchserviceagent.watchservice_agent.analytics.EventWindowAggregator;
import com.watchserviceagent.watchservice_agent.collector.FileCollectorService;
import com.watchserviceagent.watchservice_agent.collector.dto.FileAnalysisResult;
import com.watchserviceagent.watchservice_agent.common.util.SessionIdManager;
import com.watchserviceagent.watchservice_agent.watcher.dto.WatcherEventRecord;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * 클래스 이름 : WatcherService
 * 기능 : 지정된 폴더를 실시간으로 감시하여 파일 생성/수정/삭제 이벤트를 감지하고, 이벤트를 Collector로 전달한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class WatcherService {

    private final SessionIdManager sessionIdManager;
    private final FileCollectorService fileCollectorService;
    private final EventWindowAggregator eventWindowAggregator;

    private WatchService watchService;
    private Thread watcherThread;
    private volatile boolean running = false;

    private final Map<WatchKey, Path> keyDirMap = new ConcurrentHashMap<>();
    private final List<Path> watchedRoots = new ArrayList<>();

    /**
     * 함수 이름 : startWatching
     * 기능 : 단일 폴더 경로를 감시하기 시작한다. 내부적으로 startWatchingMultiple을 호출한다.
     * 매개변수 : folderPath - 감시할 폴더 경로
     * 반환값 : 없음
     * 예외 : IOException - WatchService 생성 실패 시
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public synchronized void startWatching(String folderPath) throws IOException {
        startWatchingMultiple(List.of(folderPath));
    }

    /**
     * 함수 이름 : startWatchingMultiple
     * 기능 : 여러 폴더 경로를 동시에 감시하기 시작한다. WatchService를 초기화하고 모든 하위 디렉토리를 재귀적으로 등록한다.
     * 매개변수 : folderPaths - 감시할 폴더 경로 리스트
     * 반환값 : 없음
     * 예외 : IOException - WatchService 생성 실패 시, IllegalArgumentException - 경로가 유효하지 않을 때
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public synchronized void startWatchingMultiple(List<String> folderPaths) throws IOException {
        if (running) {
            log.info("Watcher already running. ignore startWatching request. paths={}", folderPaths);
            return;
        }

        List<Path> roots = new ArrayList<>();
        for (String p : (folderPaths == null ? List.<String>of() : folderPaths)) {
            if (p == null || p.isBlank()) continue;
            Path root = Paths.get(p.trim());
            if (!Files.exists(root) || !Files.isDirectory(root)) {
                throw new IllegalArgumentException("감시할 경로가 존재하지 않거나 디렉토리가 아닙니다: " + p);
            }
            roots.add(root);
        }

        if (roots.isEmpty()) {
            throw new IllegalArgumentException("감시할 폴더가 없습니다.");
        }

        this.watchService = FileSystems.getDefault().newWatchService();
        this.keyDirMap.clear();
        this.watchedRoots.clear();
        this.watchedRoots.addAll(roots);

        for (Path root : roots) {
            registerAll(root);
        }

        running = true;
        watcherThread = new Thread(this::watchLoop, "WatcherService-Thread");
        watcherThread.start();

        log.info("Started watching roots: {}", roots);
    }

    /**
     * 함수 이름 : registerAll
     * 기능 : 지정된 경로부터 시작하여 모든 하위 디렉토리를 재귀적으로 WatchService에 등록한다.
     * 매개변수 : start - 등록을 시작할 루트 경로
     * 반환값 : 없음
     * 예외 : IOException - 파일 시스템 접근 실패 시
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private void registerAll(Path start) throws IOException {
        Files.walkFileTree(start, new HashSet<>(), Integer.MAX_VALUE, new SimpleFileVisitor<>() {
            @Override
            public FileVisitResult preVisitDirectory(Path dir, BasicFileAttributes attrs) throws IOException {
                register(dir);
                return FileVisitResult.CONTINUE;
            }
        });
    }

    /**
     * 함수 이름 : register
     * 기능 : 단일 디렉토리를 WatchService에 등록하여 CREATE/MODIFY/DELETE 이벤트를 감지할 수 있도록 한다.
     * 매개변수 : dir - 등록할 디렉토리 경로
     * 반환값 : 없음
     * 예외 : IOException - 디렉토리 등록 실패 시
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private void register(Path dir) throws IOException {
        WatchKey key = dir.register(
                watchService,
                StandardWatchEventKinds.ENTRY_CREATE,
                StandardWatchEventKinds.ENTRY_MODIFY,
                StandardWatchEventKinds.ENTRY_DELETE
        );
        keyDirMap.put(key, dir);
        log.debug("Registered directory for watching: {}", dir);
    }

    /**
     * 함수 이름 : stopWatching
     * 기능 : 파일 감시를 중지하고 모든 리소스를 정리한다. 남은 윈도우 이벤트를 flush한다.
     * 매개변수 : 없음
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public synchronized void stopWatching() {
        if (!running) {
            log.info("Watcher is not running. ignore stopWatching request.");
            return;
        }

        running = false;

        if (watchService != null) {
            try {
                watchService.close();
            } catch (IOException e) {
                log.warn("Failed to close WatchService", e);
            }
        }

        if (watcherThread != null) {
            watcherThread.interrupt();
            try {
                watcherThread.join(3000);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                log.warn("Interrupted while waiting for watcherThread to join", e);
            }
        }

        keyDirMap.clear();
        watchedRoots.clear();

        // 종료 시 남은 윈도우 flush
        eventWindowAggregator.flushIfNeeded();

        log.info("Stopped watching.");
    }

    /**
     * 함수 이름 : watchLoop
     * 기능 : WatchService에서 이벤트를 지속적으로 수신하여 처리하는 메인 루프. 각 이벤트를 Collector와 Analytics로 전달한다.
     * 매개변수 : 없음
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private void watchLoop() {
        String ownerKey = sessionIdManager.getSessionId();
        log.info("Watcher loop started. ownerKey={}", ownerKey);

        try {
            while (running) {
                WatchKey key;
                try {
                    key = watchService.take();
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    log.info("Watcher loop interrupted");
                    break;
                } catch (ClosedWatchServiceException e) {
                    log.info("WatchService has been closed. stop watcher loop.");
                    break;
                }

                Path dir = keyDirMap.get(key);
                if (dir == null) {
                    boolean valid = key.reset();
                    if (!valid) keyDirMap.remove(key);
                    continue;
                }

                for (WatchEvent<?> event : key.pollEvents()) {
                    WatchEvent.Kind<?> kind = event.kind();

                    if (kind == StandardWatchEventKinds.OVERFLOW) {
                        log.warn("WatchService overflow event occurred. some events may have been lost.");
                        continue;
                    }

                    @SuppressWarnings("unchecked")
                    WatchEvent<Path> ev = (WatchEvent<Path>) event;
                    Path name = ev.context();
                    Path child = dir.resolve(name);

                    String eventType = mapKindToEventType(kind);

                    WatcherEventRecord record = WatcherEventRecord.builder()
                            .ownerKey(ownerKey)
                            .eventType(eventType)
                            .path(child.toAbsolutePath().toString())
                            .eventTimeMs(System.currentTimeMillis())
                            .build();

                    FileAnalysisResult analysisResult = fileCollectorService.analyze(record);

                    // ✅ 윈도우 집계+AI+로그 저장
                    eventWindowAggregator.onFileAnalysisResult(analysisResult);

                    // 새 폴더 생성되면 자동 등록
                    if (kind == StandardWatchEventKinds.ENTRY_CREATE) {
                        if (Files.isDirectory(child)) {
                            try {
                                registerAll(child);
                            } catch (IOException e) {
                                log.warn("Failed to register sub directory: {}", child, e);
                            }
                        }
                    }
                }

                boolean valid = key.reset();
                if (!valid) {
                    keyDirMap.remove(key);
                    if (keyDirMap.isEmpty()) {
                        log.info("No directories are being watched anymore. stopping watchLoop.");
                        break;
                    }
                }
            }
        } catch (Exception e) {
            log.error("Unexpected error in watcher loop", e);
        } finally {
            running = false;
            log.info("Watcher loop finished.");
        }
    }

    /**
     * 함수 이름 : mapKindToEventType
     * 기능 : WatchService의 이벤트 종류를 내부 이벤트 타입 문자열로 변환한다.
     * 매개변수 : kind - WatchService 이벤트 종류
     * 반환값 : "CREATE", "MODIFY", "DELETE", 또는 "UNKNOWN"
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private String mapKindToEventType(WatchEvent.Kind<?> kind) {
        if (kind == StandardWatchEventKinds.ENTRY_CREATE) return "CREATE";
        if (kind == StandardWatchEventKinds.ENTRY_MODIFY) return "MODIFY";
        if (kind == StandardWatchEventKinds.ENTRY_DELETE) return "DELETE";
        return "UNKNOWN";
    }

    /**
     * 함수 이름 : isRunning
     * 기능 : 현재 파일 감시가 실행 중인지 여부를 반환한다.
     * 매개변수 : 없음
     * 반환값 : true면 실행 중, false면 중지됨
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public boolean isRunning() {
        return running;
    }
}
