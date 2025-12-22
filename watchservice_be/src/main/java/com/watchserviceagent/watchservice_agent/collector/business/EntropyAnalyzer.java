package com.watchserviceagent.watchservice_agent.collector.business;

import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 클래스 이름 : EntropyAnalyzer
 * 기능 : 파일의 Shannon 엔트로피를 계산하는 유틸리티 컴포넌트. 암호화된 파일 탐지에 사용된다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Component
public class EntropyAnalyzer {

    /**
     * 함수 이름 : computeSampleEntropy
     * 기능 : 파일에서 최대 maxBytes 만큼 읽어 Shannon 엔트로피를 계산한다. 바이트 값 0~255의 출현 빈도를 세어 계산한다.
     * 매개변수 : path - 대상 파일 경로, maxBytes - 최대 샘플 바이트 수 (예: 4096)
     * 반환값 : double - 엔트로피 값 (0 이상, 대략 최대 8 근처)
     * 예외 : IOException - 파일 읽기 실패 시
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public double computeSampleEntropy(Path path, int maxBytes) throws IOException {
        if (maxBytes <= 0) {
            throw new IllegalArgumentException("maxBytes must be positive");
        }

        int[] freq = new int[256];
        int total = 0;

        try (InputStream in = Files.newInputStream(path)) {
            byte[] buffer = new byte[1024];
            while (total < maxBytes) {
                int remaining = maxBytes - total;
                int toRead = Math.min(buffer.length, remaining);
                int read = in.read(buffer, 0, toRead);
                if (read == -1) {
                    break; // EOF
                }
                for (int i = 0; i < read; i++) {
                    int b = buffer[i] & 0xFF;
                    freq[b]++;
                }
                total += read;
            }
        }

        if (total == 0) {
            // 빈 파일 등의 경우 엔트로피 0
            return 0.0;
        }

        double entropy = 0.0;
        for (int count : freq) {
            if (count == 0) continue;
            double p = (double) count / (double) total;
            entropy -= p * (log2(p));
        }
        return entropy;
    }

    /**
     * 함수 이름 : computeSampleEntropy
     * 기능 : 기본 샘플 크기(4096 bytes)로 엔트로피를 계산한다.
     * 매개변수 : path - 대상 파일 경로
     * 반환값 : double - 엔트로피 값
     * 예외 : IOException - 파일 읽기 실패 시
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public double computeSampleEntropy(Path path) throws IOException {
        return computeSampleEntropy(path, 4096);
    }

    /**
     * 함수 이름 : computeEntropy
     * 기능 : 메모리 상의 바이트 배열 전체에 대해 엔트로피를 계산한다.
     * 매개변수 : data - 바이트 배열
     * 반환값 : double - 엔트로피 값
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    public double computeEntropy(byte[] data) {
        if (data == null || data.length == 0) {
            return 0.0;
        }

        int[] freq = new int[256];
        for (byte b : data) {
            freq[b & 0xFF]++;
        }

        int total = data.length;
        double entropy = 0.0;

        for (int count : freq) {
            if (count == 0) continue;
            double p = (double) count / (double) total;
            entropy -= p * (log2(p));
        }
        return entropy;
    }

    /**
     * 함수 이름 : log2
     * 기능 : 밑이 2인 로그를 계산한다.
     * 매개변수 : x - 입력 값
     * 반환값 : double - log2(x)
     * 작성 날짜 : 2025/12/17
     * 작성자 : 시스템
     */
    private double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }
}
