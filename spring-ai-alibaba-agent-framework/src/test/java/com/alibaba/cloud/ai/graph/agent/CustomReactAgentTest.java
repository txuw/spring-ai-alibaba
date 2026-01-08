package com.alibaba.cloud.ai.graph.agent;

import com.alibaba.cloud.ai.dashscope.api.DashScopeApi;
import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatModel;
import com.alibaba.cloud.ai.graph.StateGraph;
import com.alibaba.cloud.ai.graph.agent.flow.agent.FlowAgent;
import com.alibaba.cloud.ai.graph.agent.flow.builder.FlowAgentBuilder;
import com.alibaba.cloud.ai.graph.agent.flow.builder.FlowGraphBuilder;
import com.alibaba.cloud.ai.graph.agent.flow.enums.FlowAgentEnum;
import com.alibaba.cloud.ai.graph.exception.GraphStateException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.ai.chat.model.ChatModel;

import java.util.List;
import java.util.Map;

/**
 * @author : zhengyuchao
 * @date : 2026/1/8
 */

@EnabledIfEnvironmentVariable(named = "AI_DASHSCOPE_API_KEY", matches = ".+")
public class CustomReactAgentTest {

    private ChatModel chatModel;

    @BeforeEach
    void setUp() {
        // Create DashScopeApi instance using the API key from environment variable
        DashScopeApi dashScopeApi = DashScopeApi.builder().apiKey(System.getenv("AI_DASHSCOPE_API_KEY")).build();

        // Create DashScope ChatModel instance
        this.chatModel = DashScopeChatModel.builder().dashScopeApi(dashScopeApi).build();
    }

    static class ConditionalAgent extends FlowAgent {

        private final Map<String, Agent> condition;

        protected ConditionalAgent(ConditionalAgentBuilder builder) throws GraphStateException {
            super(builder.name, builder.description, builder.compileConfig,
                    builder.condition.values().stream().toList());
            this.condition = builder.condition;
        }

        @Override
        protected StateGraph buildSpecificGraph(FlowGraphBuilder.FlowGraphConfig config) throws GraphStateException {
            // 使用 FlowGraphBuilder 构建自定义图结构
            config.conditionalAgents(condition);
            return FlowGraphBuilder.buildGraph(
                    FlowAgentEnum.CONDITIONAL.getType(),
                    config
            );
        }

        public static class ConditionalAgentBuilder
                extends FlowAgentBuilder<ConditionalAgent, ConditionalAgentBuilder> {

            private Map<String, Agent> condition;
            public ConditionalAgentBuilder condition(Map<String, Agent> condition) {
                this.condition = condition;
                return this;
            }


            @Override
            protected ConditionalAgentBuilder self() {
                return this;
            }

            @Override
            public ConditionalAgent doBuild() {
                if (condition == null) {
                    throw new IllegalStateException(
                            "Condition must be set");
                }
                try {
                    return new ConditionalAgent(this);
                } catch (GraphStateException e) {
                    throw new RuntimeException(e);
                }
            }
        }
    }
}
