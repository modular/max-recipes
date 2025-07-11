# Learn How to Build AI Agents with DeepSeek-R1, AutoGen and MAX

This recipe demonstrates how to build AI agents using:

* [`DeepSeek-R1-Distill-Qwen-7B`](https://builds.modular.com/models/DeepSeek-R1-Distill-Qwen/7B) model that runs on GPU
* [AutoGen](https://microsoft.github.io/autogen/stable/) framework for multi-agent conversations
* [MAX](https://docs.modular.com/max/serve/) for efficient model serving and inference
* [Rich](https://rich.readthedocs.io/en/stable/introduction.html) Python library for beautiful terminal interfaces

We'll create two example applications that showcase:

* A conversational AI assistant with thought process visibility leveraging DeepSeek thinking process
* A collaborative screenplay development system with multiple specialized agents

The patterns demonstrated here can be adapted for various agent-based applications like:

* **Customer service automation**
* **Educational tutoring systems**
* **Creative writing assistants**
* **Technical support agents**
* **Research assistants**

## Requirements

Please make sure your system meets our [system requirements](https://docs.modular.com/max/get-started).

To proceed, ensure you have the `pixi` CLI installed:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

...and updated to the latest version:

```bash
pixi self-update
```

### GPU requirements

### Important: GPU requirements

This recipe requires a GPU with CUDA 12.5 support. Recommended GPUs:

* NVIDIA H100 / H200, A100, A40, L40

## Quick start

1. Download the code for this recipe:

    ```bash
    git clone https://github.com/modularml/max-recipes.git
    cd max-recipes/deepseek-qwen-autogen-agent
    ```

2. Run the MAX server via in a terminal:

    **Make sure the port `8010` is available. You can adjust the port settings in [pyproject.toml](./pyproject.toml).**

    ```bash
    pixi run server
    ```

3. In a new terminal, run either example:

    * For the chat agent:

        ```bash
        pixi run chat_agent
        ```

    * For the screenplay development team:

        ```bash
        pixi run screenplay_agents
        ```

The agents will be ready when you see the welcome message in your terminal.

## Quick demos

### Chat agent with visible thinking

<img src="https://cdn.githubraw.com/modular/devrel-extras/main/gifs/chat_agent.gif" alt="Chat interface" width="100%" style="max-width: 800px;">

*Demo shows:*

* Starting a conversation with the AI
* AI's thinking process displayed in yellow panels
* Final responses in green panels
* Multiple turns of natural conversation
* Example of complex reasoning task

### Screenplay development team

<img src="https://cdn.githubraw.com/modular/devrel-extras/main/gifs/screenplay_agents.gif" alt="Screenplay agents" width="100%" style="max-width: 800px;">

*Demo shows:*

* User providing initial scene idea
* Screenwriter creating first draft
* Story Critic analyzing and improving
* Dialogue Expert polishing the scene
* Full collaborative workflow between agents

## Technical deep dive

### Single agent chat implementation

#### 1. Agent configuration

```python
client = OpenAIChatCompletionClient(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    base_url=LLM_SERVER_URL,
    api_key=LLM_API_KEY,
    model_info={
        "name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "family": "deepseek",
        "pricing": {"prompt": 0.0, "completion": 0.0},
    },
    temperature=0.7,  # Adjust for more/less creative responses
    max_tokens=4096,  # Adjust for longer conversation
)

assistant = AssistantAgent(
    name="assistant",
    model_client=client,
    system_message="You are a helpful assistant.",
)
```

Key features:

* Uses DeepSeek-R1 model through MAX
* Configurable temperature for response creativity
* Adjust `max_tokens` for longer conversations

#### 2. Conversation management

```python
# Keep track of conversation history
conversation_history = []

# Add user message to history
conversation_history.append(TextMessage(content=user_input, source="user"))

# Get AI response with full context
response = await assistant.on_messages(
    conversation_history,  # Pass entire history
    CancellationToken(),
)
conversation_history.append(response.chat_message)
```

Benefits:

* Maintains context across multiple turns
* Enables coherent multi-turn conversations
* Preserves conversation state

#### 3. Thinking process visibility

The agent separates its thinking from its final response:

```python
content = response.chat_message.content
parts = content.split('</think>')
if len(parts) > 1:
    thinking = parts[0].strip()
    final_answer = parts[1].strip()
else:
    thinking = "No explicit thinking process shown"
    final_answer = content.strip()
```

This allows users to see:

* The reasoning process
* Considered alternatives
* Decision-making steps

#### 4. Rich terminal interface

The chat interface uses Rich for enhanced visualization:

```python
console.print(Panel(
    Markdown(thinking),
    border_style="yellow",
    title="[bold]💭 Thinking Process[/bold]",
    title_align="left"
))

console.print(Panel(
    Markdown(final_answer),
    border_style="green",
    title="[bold]🤖 Response[/bold]",
    title_align="left"
))
```

Features:

* Color-coded panels for different content types
* Markdown rendering for formatted text
* Clear separation of thinking and responses
* Status indicators during processing

### Multi-agent screenplay development

The screenplay development system uses multiple specialized agents working together through round-robin conversation. Here's a detailed look at the implementation:

#### 1. Specialized agent definitions

```python
screenwriter = AssistantAgent(
    name="screenwriter",
    system_message="""You are an experienced screenwriter who creates engaging movie scenes and dialogue.
First think about the scene carefully. Consider:
- Setting and atmosphere
- Character development
- Plot progression
Then write a brief but vivid scene with clear stage directions and dialogue.""",
    model_client=client,
)

story_critic = AssistantAgent(
    name="story_critic",
    system_message="""You are a story development expert. Review the screenwriter's scene and think about:
- Plot coherence and dramatic tension
- Character motivations and arcs
- Theme development
Then provide:
1. An improved version of the scene
2. A list of specific improvements made and why they work better""",
    model_client=client,
)

dialogue_expert = AssistantAgent(
    name="dialogue_expert",
    system_message="""You are a dialogue specialist. Review both the original and improved scenes, then think about:
- Character voice authenticity
- Subtext and emotional depth
- Natural flow and rhythm
Then provide:
1. A final version with enhanced dialogue
2. A list of specific dialogue improvements made and their impact on the scene""",
    model_client=client,
)
```

Each agent has a specific role:

* **Screenwriter**: Creates initial scene drafts
* **Story Critic**: Improves plot and structure
* **Dialogue Expert**: Enhances character voices and interactions

#### 2. Team coordination

```python
agent_team = RoundRobinGroupChat(
    agents=[screenwriter, story_critic, dialogue_expert],
    max_turns=3,  # One turn for each agent
    speaker_selection_method="round_robin"
)
```

The round-robin chat ensures:

* Ordered turn-taking between agents
* Complete review cycle for each scene
* Collaborative improvement process

#### 3. Message processing and display

```python
async for message in stream:
    if hasattr(message, 'content') and hasattr(message, 'source'):
        content = message.content
        source = message.source

        # Split thinking and contribution
        parts = content.split('</think>')
        if len(parts) > 1:
            thinking = parts[0].strip()
            final_answer = parts[1].strip()
        else:
            thinking = "No explicit thinking process shown"
            final_answer = content.strip()

        # Display each agent's contribution
        console.print(Panel(
            Markdown(thinking),
            border_style="yellow",
            title=f"[bold]💭 {source} Thinking[/bold]",
            title_align="left"
        ))

        console.print(Panel(
            Markdown(final_answer),
            border_style="green",
            title=f"[bold]🎬 {source} Contribution[/bold]",
            title_align="left"
        ))
```

Features:

* Real-time display of each agent's process
* Clear attribution of contributions
* Separation of thinking and final output
* Rich formatting for readability

#### Example workflow

1. **Initial Scene Creation**

```
User: Write a scene about a reunion between old friends
Screenwriter: [Analyzes setting and characters]
Screenwriter: [Creates initial scene with dialogue]
```

2. **Story Review and Enhancement**

```
Story Critic: [Evaluates dramatic tension]
Story Critic: [Suggests structural improvements]
```

3. **Dialogue Polish**

```
Dialogue Expert: [Analyzes character voices]
Dialogue Expert: [Enhances dialogue authenticity]
```

#### Customization options

You can adapt the screenplay system for different creative tasks:

1. **Add Specialized Agents**

```python
technical_advisor = AssistantAgent(
    name="technical_advisor",
    system_message="""You ensure accuracy in specialized scenes...""",
    model_client=client
)
```

2. **Modify Review Cycles**

```python
agent_team = RoundRobinGroupChat(
    agents=[...],
    max_turns=4,  # Add more revision cycles
)
```

3. **Enhance Agent Specialties**

```python
system_message="""You are a genre specialist focusing on:
- Genre conventions
- Typical plot structures
- Character archetypes
..."""
```

## Customizing for your use case

This recipe can be adapted for various applications:

1. **Educational Tutoring**
   * Add agents for different subjects
   * Implement knowledge assessment
   * Create personalized learning paths

2. **Technical Support**
   * Add agents for different technical domains
   * Implement troubleshooting workflows
   * Create solution verification steps

3. **Research Assistant**
   * Add agents for literature review
   * Implement data analysis
   * Create report generation

## Troubleshooting

Common issues and solutions:

1. **Server Connection Issues**
   * Ensure MAX is running (`pixi run server`)
   * Check if the default port `8010` is available
   * Verify network connectivity

2. **Agent Response Issues**
   * Check system messages for clarity
   * Adjust max_turns for multi-agent scenarios
   * Verify conversation history handling

3. **Performance Issues**
   * Monitor GPU memory usage
   * Adjust batch sizes if needed
   * Consider reducing conversation history length

## Conclusion

This recipe demonstrates how to:

* Build single and multi-agent systems with AutoGen
* Use DeepSeek-R1 and examine its thinking process
* Create beautiful terminal interfaces with Rich Python library
* Implement robust error handling
* Enable collaborative agent interactions

The patterns shown here provide a foundation for building your own agent-based applications.

## Next steps

* Deploy your agents on [AWS, GCP or Azure](https://docs.modular.com/max/tutorials/max-serve-local-to-cloud/)
* Explore [MAX documentation](https://docs.modular.com/max/) for more features
* Join our [Modular Forum](https://forum.modular.com/) and [Discord community](https://discord.gg/modular)

Share your agent projects with us using `#ModularAI` on social media!
