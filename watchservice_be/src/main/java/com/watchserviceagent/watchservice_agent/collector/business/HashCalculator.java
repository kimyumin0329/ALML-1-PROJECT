package com.watchserviceagent.watchservice_agent.collector.business;

import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/**
 * 파일의 SHA-256 해시를 계산하는 유틸.
 *
 * - 파일 내용을 스트림으로 읽으면서 SHA-256 Digest를 업데이트한다.
 * - 최종 결과를 16진수 문자열(hex)로 반환한다.
 */
@Slf4j
@Component
public class HashCalculator {

    private static final int BUFFER_SIZE = 8192;

    /**
     * 주어진 파일의 SHA-256 해시를 계산한다.
     *
     * @param path 해시를 계산할 파일 경로
     * @return SHA-256 해시(16진수 문자열), 파일이 없거나 오류시 null
     */
    public String calculateSha256(Path path) {
        if (path == null || !Files.isRegularFile(path)) {
            log.warn("[HashCalculator] 유효하지 않은 파일 경로: {}", path);
            return null;
        }

        MessageDigest digest;
        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException e) {
            // 이론상 발생하지 않음 (JVM에 SHA-256 기본 제공)
            log.error("[HashCalculator] SHA-256 알고리즘을 사용할 수 없습니다.", e);
            return null;
        }

        try (InputStream in = Files.newInputStream(path)) {
            byte[] buffer = new byte[BUFFER_SIZE];
            int read;
            while ((read = in.read(buffer)) != -1) {
                digest.update(buffer, 0, read);
            }
        } catch (IOException e) {
            log.warn("[HashCalculator] 파일 읽기 중 예외 발생: {}", path, e);
            return null;
        }

        byte[] hashBytes = digest.digest();
        return toHex(hashBytes);
    }

    /**
     * 바이트 배열을 16진수 문자열로 변환.
     *
     * @param bytes 해시 바이트 배열
     * @return 16진수 문자열
     */
    private String toHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder(bytes.length * 2);
        for (byte b : bytes) {
            // & 0xFF로 부호 제거 후, 16진수 2자리로 포맷
            sb.append(String.format("%02x", b & 0xFF));
        }
        return sb.toString();
    }
}
