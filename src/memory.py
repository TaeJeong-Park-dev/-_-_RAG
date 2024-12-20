from langchain.memory import ConversationSummaryBufferMemory

def create_memory(llm):
    return ConversationSummaryBufferMemory(
        llm=llm,
        memory_key="chat_history",
        return_messages=True,
        max_token_limit=2000,
        output_key="answer"
    )


print("메모리가 준비되었습니다.")