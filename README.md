# Dynamic Agentic Pipeline
---

![lats](../assets/lats.png)

This folder contains the code for our dynamic argentic implementation utilizing the **LATS (Language Agent Tree Search)** and various tools, allowing us to generate detailed reports on the user's query. The main pipeline is organized as follows:

1. **``Agents``**: This folder contains the code for the **Planner Agent**, code used to generate dynamic sub-agents which address different aspects of the user query, and the other agents used during the compilation of responses.
    - **``PlannerAgent.py``**: This file contains the code for the Planner Agent, which is responsible for generating the sub-agents that will be used to address the user query.
    - **``Agents.py``**: Contains the class for defining each of the agents that will be created by the Planner Agent.
    - **``Smack.py``**: This file is designed to manage and execute a set of interdependent tasks efficiently by leveraging parallel execution and handling task dependencies dynamically.
    - **``RAG_Agent.py``**: This file contains the code for the RAG Agent, which utilizes the RAG model to generate responses to the user query.
    - **``ChartGenAgent.py``**: This file contains the code for the ChartGen Agent, which is used to generate charts in the end report, based on the data extracted from the user query.
    - **``ClassifierAgent.py``**: This file contains the code for the Classifier Agent, which is used to classify the user query into different categories, based on the intent of the user.  
    
    All the other agents are used to generate the final response to the user query, compiling the responses from the sub-agents.
2. **``Agents/LATS``**: This folder contains the **implementation of LATS**, which is the architecture utilized by each of our sub-agents to do a detailed analysis of the subquery passed to them.
    - **``TreeState.py``**: Contains the TreeState class, which is used to represent the state of the decision tree in the LATS agent.
    - **``Create_graph.py``**: Generates the graph structure for the LATS agent using the StateGraph class from langgraph.
    - **``Initial_response.py``**: Generates the initial response for the LATS agent, including calling necessary tools and providing detailed answers.
    - **``generate_candidates.py``**: Contains functions to generate and expand candidate nodes for the LATS agent's decision tree, first it calls the initial response function to generate the initial response and then generates the candidate nodes, based on the evaluation of Reflection.
    - **``Reflection.py``**: Contains the reflection function, which is used to evaluate the response generated by each iteration within LATS, and determine whether to continue expanding the decision tree or generate the final response.
    - **``NewTools.py``**: Contains the code for the tools used by the LATS agent to generate the final response.
    - **``Solve_subquery.py``**: Contains the code for the LATS agent to solve the subquery passed to it, by generating the decision tree and expanding the candidate nodes.

3. **``Tools``**: This folder contains the description of the tools in the [info.json](./Tools/info.json), this JSON file will be used by the Planner Agent to assign tools to each of the sub-agents.
4. **``Change.py``**: This python file is used to verbose the process of our pipeline, by passing the intermediate results to the UI.
5. **``main.py``**: This file is the main file for the pipeline, which is used to generate the final response to the user query by calling the Planner Agent and the sub-agents.
6. **``TopicalGuardrails.py``**: This file contains the code for the Guardrails implemented in our pipeline, to ensure the safety of the responses generated by the agents.

Following is the tree structure of the main pipeline:
```
pipeline
├── Agents
│   ├── PlannerAgent.py
│   ├── Agents.py
│   ├── Smack.py
│   ├── RAG_Agent.py
│   ├── ChartGenAgent.py
│   ├── ClassifierAgent.py
│   ├── ConciseAnsAgent.py
│   ├── conciseLatsAgent.py
│   ├── DrafterAgent.py
│   └── LATS
│       ├── TreeState.py
│       ├── Create_graph.py
│       ├── Initial_response.py
│       ├── generate_candidates.py
│       ├── Reflection.py
│       ├── NewTools.py
│       └── Solve_subquery.py
├── Tools
│   └── info.json
├── Change.py
├── main.py
└── TopicalGuardrails.py

```