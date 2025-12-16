package com.watchserviceagent.watchservice_agent.settings;

import com.watchserviceagent.watchservice_agent.settings.dto.ExceptionRuleRequest;
import com.watchserviceagent.watchservice_agent.settings.dto.ExceptionRuleResponse;
import com.watchserviceagent.watchservice_agent.settings.dto.WatchedFolderRequest;
import com.watchserviceagent.watchservice_agent.settings.dto.WatchedFolderResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import javax.swing.JFileChooser;
import java.awt.EventQueue;
import java.awt.GraphicsEnvironment;
import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;

@RestController
@RequestMapping("/settings")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
public class SettingsController {

    private final SettingsService settingsService;

    // ===== 감시 폴더 =====

    @GetMapping("/folders")
    public List<WatchedFolderResponse> getWatchedFolders() {
        List<WatchedFolderResponse> list = settingsService.getWatchedFolders();
        log.info("[SettingsController] GET /settings/folders -> {}건", list.size());
        return list;
    }

    @PostMapping("/folders")
    public WatchedFolderResponse addWatchedFolder(@RequestBody WatchedFolderRequest req) {
        WatchedFolderResponse resp = settingsService.addWatchedFolder(req);
        log.info("[SettingsController] POST /settings/folders -> {}", resp);
        return resp;
    }

    @DeleteMapping("/folders/{id}")
    public void deleteWatchedFolder(@PathVariable("id") Long id) {
        settingsService.deleteWatchedFolder(id);
        log.info("[SettingsController] DELETE /settings/folders/{}", id);
    }

    /**
     * ✅ 폴더 선택 다이얼로그
     * GET /settings/folders/pick -> { "path": "C:\\Users\\..." } 또는 빈값이면 취소
     */
    @GetMapping("/folders/pick")
    public ResponseEntity<?> pickFolder() {
        try {
            if (GraphicsEnvironment.isHeadless()) {
                // headless면 다이얼로그 불가
                return ResponseEntity.status(409).body("Headless environment: cannot open folder picker");
            }

            AtomicReference<String> pickedPath = new AtomicReference<>("");

            Runnable job = () -> {
                JFileChooser chooser = new JFileChooser();
                chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
                chooser.setDialogTitle("감시 폴더 선택");
                chooser.setAcceptAllFileFilterUsed(false);

                int result = chooser.showOpenDialog(null);
                if (result == JFileChooser.APPROVE_OPTION) {
                    File selected = chooser.getSelectedFile();
                    pickedPath.set(selected != null ? selected.getAbsolutePath() : "");
                } else {
                    pickedPath.set("");
                }
            };

            // EDT 처리
            if (EventQueue.isDispatchThread()) job.run();
            else EventQueue.invokeAndWait(job);

            String path = pickedPath.get();
            return ResponseEntity.ok(Map.of("path", path == null ? "" : path));
        } catch (Exception e) {
            log.error("[SettingsController] folder pick failed", e);
            return ResponseEntity.internalServerError().body("folder pick failed: " + e.getMessage());
        }
    }

    // ===== 예외 규칙 =====

    @GetMapping("/exceptions")
    public List<ExceptionRuleResponse> getExceptionRules() {
        List<ExceptionRuleResponse> list = settingsService.getExceptionRules();
        log.info("[SettingsController] GET /settings/exceptions -> {}건", list.size());
        return list;
    }

    @PostMapping("/exceptions")
    public ExceptionRuleResponse addExceptionRule(@RequestBody ExceptionRuleRequest req) {
        ExceptionRuleResponse resp = settingsService.addExceptionRule(req);
        log.info("[SettingsController] POST /settings/exceptions -> {}", resp);
        return resp;
    }

    @DeleteMapping("/exceptions/{id}")
    public void deleteExceptionRule(@PathVariable("id") Long id) {
        settingsService.deleteExceptionRule(id);
        log.info("[SettingsController] DELETE /settings/exceptions/{}", id);
    }
}
