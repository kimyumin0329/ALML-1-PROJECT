package com.watchserviceagent.watchservice_agent.collector.business;

import org.springframework.stereotype.Component;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 파일의 Shannon 엔트로피를 계산하는 유틸성 컴포넌트.
 *
 * - computeSampleEntropy(Path): 파일에서 최대 maxBytes 만큼 읽어서 샘플 엔트로피를 계산.
 * - computeEntropy(byte[]): 메모리 상의 바이트 배열에 대해 엔트로피 계산.
 */
@Component
public class EntropyAnalyzer {

    /**
     * 파일에서 최대 maxBytes 만큼 읽어 Shannon 엔트로피를 계산한다.
     *
     * - 바이트 값 0~255의 출현 빈도를 세고,
     * - p_i = count_i / N 에 대해
     *   H = - Σ p_i * log2(p_i) 를 계산한다.
     *
     * @param path     대상 파일 경로
     * @param maxBytes 최대 샘플 바이트 수 (예: 4096)
     * @return 엔트로피 값 (0 이상, 대략 최대 8 근처)
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
     * 기본 샘플 크기(4096 bytes)로 엔트로피 계산.
     */
    public double computeSampleEntropy(Path path) throws IOException {
        return computeSampleEntropy(path, 4096);
    }

    /**
     * 메모리 상의 바이트 배열 전체에 대해 엔트로피 계산.
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

    private double log2(double x) {
        return Math.log(x) / Math.log(2.0);
    }
}
