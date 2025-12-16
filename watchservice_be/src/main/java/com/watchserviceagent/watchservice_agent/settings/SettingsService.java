package com.watchserviceagent.watchservice_agent.settings;

import com.watchserviceagent.watchservice_agent.common.util.SessionIdManager;
import com.watchserviceagent.watchservice_agent.settings.domain.WatchedFolder;
import com.watchserviceagent.watchservice_agent.settings.domain.ExceptionRule;
import com.watchserviceagent.watchservice_agent.settings.dto.WatchedFolderRequest;
import com.watchserviceagent.watchservice_agent.settings.dto.WatchedFolderResponse;
import com.watchserviceagent.watchservice_agent.settings.dto.ExceptionRuleRequest;
import com.watchserviceagent.watchservice_agent.settings.dto.ExceptionRuleResponse;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

/**
 * 설정(감시 폴더, 예외 규칙) 비즈니스 로직.
 *
 * - SessionIdManager 를 통해 ownerKey 를 자동으로 부여/필터링한다.
 */
@Service
@RequiredArgsConstructor
@Slf4j
public class SettingsService {

    private final SessionIdManager sessionIdManager;
    private final SettingsRepository settingsRepository;

    // ===== 감시 폴더 =====

    public List<WatchedFolderResponse> getWatchedFolders() {
        String ownerKey = sessionIdManager.getSessionId();
        List<WatchedFolder> list = settingsRepository.findWatchedFolders(ownerKey);
        return list.stream()
                .map(WatchedFolderResponse::from)
                .collect(Collectors.toList());
    }

    public WatchedFolderResponse addWatchedFolder(WatchedFolderRequest req) {
        String ownerKey = sessionIdManager.getSessionId();

        String name = (req.getName() == null || req.getName().isBlank())
                ? req.getPath()
                : req.getName();

        WatchedFolder folder = settingsRepository.insertWatchedFolder(ownerKey, name, req.getPath());
        log.info("[SettingsService] 감시 폴더 추가: {}", folder);
        return WatchedFolderResponse.from(folder);
    }

    public void deleteWatchedFolder(Long id) {
        String ownerKey = sessionIdManager.getSessionId();
        settingsRepository.deleteWatchedFolder(ownerKey, id);
        log.info("[SettingsService] 감시 폴더 삭제: id={}", id);
    }

    // ===== 예외 규칙 =====

    public List<ExceptionRuleResponse> getExceptionRules() {
        String ownerKey = sessionIdManager.getSessionId();
        List<ExceptionRule> list = settingsRepository.findExceptionRules(ownerKey);
        return list.stream()
                .map(ExceptionRuleResponse::from)
                .collect(Collectors.toList());
    }

    public ExceptionRuleResponse addExceptionRule(ExceptionRuleRequest req) {
        String ownerKey = sessionIdManager.getSessionId();

        String type = (req.getType() == null || req.getType().isBlank())
                ? "PATH"
                : req.getType().toUpperCase();

        ExceptionRule rule = settingsRepository.insertExceptionRule(
                ownerKey,
                type,
                req.getPattern(),
                req.getMemo()
        );
        log.info("[SettingsService] 예외 규칙 추가: {}", rule);
        return ExceptionRuleResponse.from(rule);
    }

    public void deleteExceptionRule(Long id) {
        String ownerKey = sessionIdManager.getSessionId();
        settingsRepository.deleteExceptionRule(ownerKey, id);
        log.info("[SettingsService] 예외 규칙 삭제: id={}", id);
    }
}
