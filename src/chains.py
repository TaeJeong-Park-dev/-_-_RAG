from langchain.chains import ConversationalRetrievalChain

def create_qa_chain(llm, retriever, memory, prompt):
    """
    Conversational Retrieval Chain 생성
    """
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        return_generated_question=False,
        verbose=True
    )

print("QA 체인이 준비되었습니다.")