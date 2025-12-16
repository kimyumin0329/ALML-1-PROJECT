// src/main/java/com/watchserviceagent/watchservice_agent/collector/snapshot/SnapshotConfig.java
package com.watchserviceagent.watchservice_agent.collector.snapshot;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class SnapshotConfig {

    @Bean
    public FileSnapshotStore fileSnapshotStore() {
        return new FileSnapshotStore();
    }
}
