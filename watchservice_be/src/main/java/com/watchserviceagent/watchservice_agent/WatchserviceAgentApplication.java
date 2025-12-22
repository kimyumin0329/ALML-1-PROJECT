package com.watchserviceagent.watchservice_agent;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * 클래스 이름 : WatchserviceAgentApplication
 * 기능 : Spring Boot 애플리케이션의 메인 엔트리 포인트. 파일 감시 및 랜섬웨어 탐지 에이전트를 시작한다.
 * 작성 날짜 : 2025/12/17
 * 작성자 : 시스템
 */
@Slf4j
@SpringBootApplication
public class WatchserviceAgentApplication {

	/**
	 * 함수 이름 : main
	 * 기능 : 애플리케이션을 시작하는 메인 메서드. Spring Boot 컨텍스트를 초기화하고 서버를 실행한다.
	 *        GUI 폴더 선택 다이얼로그(JFileChooser)를 항상 사용할 수 있도록
	 *        JVM 레벨의 java.awt.headless 시스템 속성을 false 로 강제 설정한다.
	 * 매개변수 : args - 명령줄 인수
	 * 반환값 : 없음
	 * 작성 날짜 : 2025/12/17
	 * 작성자 : 시스템
	 */
	public static void main(String[] args) {
		// Swing 다이얼로그를 사용하기 위해 JVM 전역 headless 모드를 비활성화
		System.setProperty("java.awt.headless", "false");

		log.info("Watchservice Agent 애플리케이션 시작");
		log.info("[Test] - 2");
		log.info("java.awt.headless 시스템 속성 = {}", System.getProperty("java.awt.headless"));

		SpringApplication.run(WatchserviceAgentApplication.class, args);
	}
}
