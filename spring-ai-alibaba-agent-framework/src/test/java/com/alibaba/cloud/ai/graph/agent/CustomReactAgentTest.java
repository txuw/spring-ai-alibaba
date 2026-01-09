package com.alibaba.cloud.ai.graph.agent;

import com.alibaba.cloud.ai.dashscope.api.DashScopeApi;
import com.alibaba.cloud.ai.dashscope.chat.DashScopeChatModel;
import com.alibaba.cloud.ai.graph.OverAllState;
import com.alibaba.cloud.ai.graph.StateGraph;
import com.alibaba.cloud.ai.graph.agent.flow.agent.FlowAgent;
import com.alibaba.cloud.ai.graph.agent.flow.builder.FlowAgentBuilder;
import com.alibaba.cloud.ai.graph.agent.flow.builder.FlowGraphBuilder;
import com.alibaba.cloud.ai.graph.agent.flow.node.ConditionEvaluatorAction;
import com.alibaba.cloud.ai.graph.agent.flow.node.TransparentNode;
import com.alibaba.cloud.ai.graph.exception.GraphStateException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatModel;

import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

import static com.alibaba.cloud.ai.graph.StateGraph.END;
import static com.alibaba.cloud.ai.graph.StateGraph.START;
import static com.alibaba.cloud.ai.graph.action.AsyncNodeAction.node_async;
import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assertions.fail;

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
            super(builder.name, builder.description, builder.compileConfig, builder.condition.values().stream().toList(),builder.stateSerializer,builder.executor);
            this.condition = builder.condition;
        }

        public static ConditionalAgentBuilder builder() {
            return new ConditionalAgentBuilder();
        }

        @Override
        protected StateGraph buildSpecificGraph(FlowGraphBuilder.FlowGraphConfig config) throws GraphStateException {

            // 1. 初始化 Graph
            StateGraph graph = new StateGraph();

            // 2. 添加根节点（透明节点作为入口）
            graph.addNode(this.name(), node_async(new TransparentNode()));
            graph.addEdge(START, this.name());

            // 3. 添加自定义条件评估节点
            String conditionNodeName = this.name() + "_condition";
            graph.addNode(conditionNodeName, node_async(state -> {
                String input = state.value("input", "").toString();
                String result = "default";

                // 简单的规则匹配
                if (input.contains("紧急")) {
                    result = "紧急";
                } else if (input.contains("常规") ) {
                    result = "常规";
                }

                // 返回结果供后续 ConditionEvaluatorAction 使用
                return Map.of("_condition_result", result);
            }));

            // 连接根节点到条件节点
            graph.addEdge(this.name(), conditionNodeName);

            // 4. 添加子 Agent 节点并构建路由表
            Map<String, String> routingMap = new HashMap<>();
            this.condition.forEach((key, agent) -> {
                // 将子 Agent 的 Graph 作为节点加入
                try {
                    graph.addNode(agent.name(), agent.getGraph());
                    // 子 Agent 执行完后指向结束
                    graph.addEdge(agent.name(), END);
                    // 记录路由关系: 条件 Key -> Agent Name
                    routingMap.put(key, agent.name());
                } catch (GraphStateException e) {
                    throw new RuntimeException(e);
                }
            });
            // 添加默认路由
            routingMap.put("default", END);

            // 5. 添加条件边 (根据 _condition_result 的值跳转)
            graph.addConditionalEdges(conditionNodeName,
                    new ConditionEvaluatorAction(),
                    routingMap);

            return graph;
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

    @Test
    public void testReactAgent() throws Exception {

        ReactAgent urgentAgent = ReactAgent.builder()
                .name("urgent_handler")
                .model(chatModel)
                .description("处理紧急请求")
                .instruction("你需要快速响应紧急情况...")
                .outputKey("urgent_result")
                .build();

        ReactAgent normalAgent = ReactAgent.builder()
                .name("normal_handler")
                .model(chatModel)
                .description("处理常规请求")
                .instruction("你可以详细分析和处理常规请求...")
                .outputKey("normal_result")
                .build();

        ConditionalAgent agent = ConditionalAgent.builder()
                .name("priority_router")
                .description("根据紧急程度选择请求")
                .condition(Map.of("紧急", urgentAgent, "常规", normalAgent))
                .build();

        try {
            Optional<OverAllState> invoke = agent.invoke("这是一个常规请求，需要你帮我写一首关于秋天的现代诗");
            OverAllState state = invoke.get();
            StringBuilder output = new StringBuilder();
            // 访问各个Agent的输出
            state.value("urgent_result", AssistantMessage.class).ifPresent(r -> {
                output.append(r.getText());
            });
            state.value("normal_result", AssistantMessage.class).ifPresent(r -> {
                output.append(r.getText());
            });

            System.out.println(output.toString());
        } catch (java.util.concurrent.CompletionException e) {
            e.printStackTrace();
            fail("ReactAgent execution failed: " + e.getMessage());
        }
    }
}
