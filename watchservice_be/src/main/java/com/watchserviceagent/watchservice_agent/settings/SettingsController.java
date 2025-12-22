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

/**
 * 클래스 이름 : SettingsController
 * 기능 : 감시 폴더 및 예외 규칙 설정을 관리하는 REST API 엔드포인트를 제공한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@RestController
@RequestMapping("/settings")
@RequiredArgsConstructor
@Slf4j
@CrossOrigin(origins = {"http://localhost:3000", "http://localhost:5173"})
public class SettingsController {

    private final SettingsService settingsService;

    /**
     * 함수 이름 : getWatchedFolders
     * 기능 : 등록된 감시 폴더 목록을 조회한다.
     * 매개변수 : 없음
     * 반환값 : WatchedFolderResponse 리스트
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/folders")
    public List<WatchedFolderResponse> getWatchedFolders() {
        List<WatchedFolderResponse> list = settingsService.getWatchedFolders();
        log.info("[SettingsController] GET /settings/folders -> {}건", list.size());
        return list;
    }

    /**
     * 함수 이름 : addWatchedFolder
     * 기능 : 새로운 감시 폴더를 추가한다.
     * 매개변수 : req - 감시 폴더 요청 객체
     * 반환값 : WatchedFolderResponse - 추가된 감시 폴더 정보
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostMapping("/folders")
    public WatchedFolderResponse addWatchedFolder(@RequestBody WatchedFolderRequest req) {
        WatchedFolderResponse resp = settingsService.addWatchedFolder(req);
        log.info("[SettingsController] POST /settings/folders -> {}", resp);
        return resp;
    }

    /**
     * 함수 이름 : deleteWatchedFolder
     * 기능 : 감시 폴더를 삭제한다.
     * 매개변수 : id - 삭제할 감시 폴더 ID
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @DeleteMapping("/folders/{id}")
    public void deleteWatchedFolder(@PathVariable("id") Long id) {
        settingsService.deleteWatchedFolder(id);
        log.info("[SettingsController] DELETE /settings/folders/{}", id);
    }

    /**
     * 함수 이름 : pickFolder
     * 기능 : GUI 폴더 선택 다이얼로그를 열어 사용자가 폴더를 선택할 수 있게 한다. (headless 환경에서는 사용 불가)
     * 매개변수 : 없음
     * 반환값 : ResponseEntity - 선택된 폴더 경로 또는 에러 메시지
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/folders/pick")
    public ResponseEntity<?> pickFolder() {
        try {
            // 혹시라도 JVM이 headless 로 떠 있으면 바로 에러 응답 (예외 대신)
            if (GraphicsEnvironment.isHeadless()) {
                log.error("[SettingsController] 현재 JVM이 headless 모드입니다. 폴더 선택 다이얼로그를 열 수 없습니다.");
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

    /**
     * 함수 이름 : getExceptionRules
     * 기능 : 등록된 예외 규칙 목록을 조회한다.
     * 매개변수 : 없음
     * 반환값 : ExceptionRuleResponse 리스트
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @GetMapping("/exceptions")
    public List<ExceptionRuleResponse> getExceptionRules() {
        List<ExceptionRuleResponse> list = settingsService.getExceptionRules();
        log.info("[SettingsController] GET /settings/exceptions -> {}건", list.size());
        return list;
    }

    /**
     * 함수 이름 : addExceptionRule
     * 기능 : 새로운 예외 규칙을 추가한다.
     * 매개변수 : req - 예외 규칙 요청 객체
     * 반환값 : ExceptionRuleResponse - 추가된 예외 규칙 정보
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @PostMapping("/exceptions")
    public ExceptionRuleResponse addExceptionRule(@RequestBody ExceptionRuleRequest req) {
        ExceptionRuleResponse resp = settingsService.addExceptionRule(req);
        log.info("[SettingsController] POST /settings/exceptions -> {}", resp);
        return resp;
    }

    /**
     * 함수 이름 : deleteExceptionRule
     * 기능 : 예외 규칙을 삭제한다.
     * 매개변수 : id - 삭제할 예외 규칙 ID
     * 반환값 : 없음
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    @DeleteMapping("/exceptions/{id}")
    public void deleteExceptionRule(@PathVariable("id") Long id) {
        settingsService.deleteExceptionRule(id);
        log.info("[SettingsController] DELETE /settings/exceptions/{}", id);
    }
}
