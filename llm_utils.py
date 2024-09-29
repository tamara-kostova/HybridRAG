from mistralai import Mistral


def run_mistral(
    client: Mistral, user_message: str, prompt: str, model="mistral-large-latest"
):
    messages = [{"role": "user", "content": prompt + user_message}]
    chat_response = client.chat.complete(model=model, messages=messages)
    return chat_response.choices[0].message.content


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
