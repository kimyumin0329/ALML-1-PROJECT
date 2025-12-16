package com.watchserviceagent.watchservice_agent;

import lombok.extern.slf4j.Slf4j;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import java.awt.*;

@Slf4j
@SpringBootApplication
public class WatchserviceAgentApplication {

	public static void main(String[] args) {
		log.info("[Test] - 6");
		System.out.println("java.awt.headless=" + System.getProperty("java.awt.headless"));
		System.out.println("isHeadless=" + GraphicsEnvironment.isHeadless());

		SpringApplication.run(WatchserviceAgentApplication.class, args);
	}
}
