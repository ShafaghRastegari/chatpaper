import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.prebuilt import create_react_agent
from typing import List

from src.agent.tools import search_papers, compare_papers, generate_literature_review

SYSTEM_PROMPT = """You are ChatPaper, an expert AI research assistant. You have access to a database of research papers.

CRITICAL RULES — NEVER BREAK THESE:
1. You MUST call search_papers tool FIRST before answering ANY question. No exceptions.
2. NEVER say you don't have enough information before searching. Always search first.
3. NEVER ask the user for clarification before searching. Search with what you have.
4. Base your answer ONLY on what the search_papers tool returns.
5. If the tool returns no relevant results, then say the paper does not contain that information.

Your tools:
- search_papers: search the indexed papers — call this IMMEDIATELY for every question
- compare_papers: compare topics across multiple papers
- generate_literature_review: write a literature review from the papers

Answer format:
- Be concise and specific
- Always mention which paper and page your answer comes from
- Use bullet points for lists
"""


class ChatPaperAgent:

    def __init__(self):
        print("Initializing ChatPaper agent...")

        self.llm = ChatOpenAI(
            model="anthropic/claude-3-haiku",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
            max_tokens=4096,
        )

        self.tools = [
            search_papers,
            compare_papers,
            generate_literature_review,
        ]

        self.graph = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )

        self.conversation_history: List[BaseMessage] = []

        print("Agent ready!")

    def chat(self, user_message: str) -> str:
        #Send a message and get the agent's response.

        self.conversation_history.append(
            HumanMessage(content=user_message)
        )

        try:
            result = self.graph.invoke(
                {"messages": self.conversation_history}
            )

            response_text = ""
            for message in reversed(result["messages"]):
                content = message.content

                if isinstance(content, str):
                    if content.strip():
                        response_text = content.strip()
                        break
                elif isinstance(content, list):
                    text_parts = []
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text" and block.get("text", "").strip():
                                text_parts.append(block["text"].strip())
                        elif isinstance(block, str) and block.strip():
                            text_parts.append(block.strip())
                    if text_parts:
                        response_text = " ".join(text_parts)
                        break

            if not response_text:
                response_text = "I could not generate a response. Please try again."

            self.conversation_history.append(
                AIMessage(content=response_text)
            )

            return response_text

        except Exception as e:
            error_msg = f"Agent error: {str(e)}\n\nPlease try rephrasing your question."
            return error_msg

    def reset(self) -> None:
        """
        Clear conversation history to start a fresh session.
        Call this when the user clicks "New Conversation" or
        when switching to a different set of papers.
        """
        self.conversation_history = []
        print("Conversation reset")