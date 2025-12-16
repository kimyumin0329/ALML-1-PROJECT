package com.watchserviceagent.watchservice_agent.scan;

import com.watchserviceagent.watchservice_agent.scan.dto.ScanProgressResponse;
import com.watchserviceagent.watchservice_agent.scan.dto.ScanStartRequest;
import com.watchserviceagent.watchservice_agent.scan.dto.ScanStartResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/scan")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
public class ScanController {

    private final ScanService scanService;

    // POST /scan/start  {paths:[...], autoStartWatcher:true/false} -> {scanId}
    @PostMapping("/start")
    public ScanStartResponse start(@RequestBody ScanStartRequest req) {
        boolean auto = (req.getAutoStartWatcher() == null) ? true : req.getAutoStartWatcher();
        String id = scanService.startScan(req.getPaths(), auto);
        log.info("[ScanController] start scanId={} autoStartWatcher={}", id, auto);
        return ScanStartResponse.builder().scanId(id).build();
    }

    // POST /scan/{scanId}/pause -> progress 형태로 응답
    @PostMapping("/{scanId}/pause")
    public ScanProgressResponse pause(@PathVariable String scanId) {
        scanService.pause(scanId);
        return scanService.getProgress(scanId);
    }

    // GET /scan/{scanId}/progress
    @GetMapping("/{scanId}/progress")
    public ScanProgressResponse progress(@PathVariable String scanId) {
        return scanService.getProgress(scanId);
    }
}
