package com.watchserviceagent.watchservice_agent.scan.domain;

import lombok.Getter;

import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

@Getter
public class ScanJob {

    public enum Status {
        RUNNING,
        PAUSED,   // 사용자가 중지(일시중지 버튼) 누른 상태(= 스캔 종료)
        DONE,
        ERROR
    }

    private final String scanId;
    private final List<String> roots;

    private final AtomicLong scanned = new AtomicLong(0);
    private final AtomicLong total = new AtomicLong(0);

    private volatile String currentPath;
    private volatile Status status = Status.RUNNING;
    private volatile String message;

    // "pause"를 실제로는 안전하게 스캔을 멈추는(stop) 용도로 처리
    private volatile boolean stopRequested = false;

    public ScanJob(String scanId, List<String> roots) {
        this.scanId = scanId;
        this.roots = roots;
        this.message = "RUNNING";
    }

    public void setTotal(long v) {
        total.set(Math.max(0, v));
    }

    public void incScanned() {
        scanned.incrementAndGet();
    }

    public int getPercent() {
        long t = total.get();
        if (t <= 0) return 0;
        long s = scanned.get();
        long pct = (s * 100L) / t;
        if (pct < 0) pct = 0;
        if (pct > 100) pct = 100;
        return (int) pct;
    }

    public void setCurrentPath(String p) {
        this.currentPath = p;
    }

    public void pause() {
        this.stopRequested = true;
        this.status = Status.PAUSED;
        this.message = "PAUSED_BY_USER";
    }

    public boolean isStopRequested() {
        return stopRequested;
    }

    public void done() {
        this.status = Status.DONE;
        this.message = "DONE";
    }

    public void error(String msg) {
        this.status = Status.ERROR;
        this.message = (msg == null || msg.isBlank()) ? "ERROR" : msg;
    }
}
