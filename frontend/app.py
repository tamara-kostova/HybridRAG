import streamlit as st
import requests
import json
import os

API_URL = os.getenv("API_URL", "http://localhost:8000")


def main():
    st.set_page_config(page_title="RAG Chat Assistant", page_icon="🤖", layout="wide")

    st.title("RAG Chat Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(
                message["content"]["response"]
                if isinstance(message["content"], dict)
                else message["content"]
            )

    if prompt := st.chat_input("What would you like to know?"):
        with st.chat_message("user"):
            st.write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking... Please wait..."):
                try:
                    form_data = {"question": prompt}
                    response = requests.post(f"{API_URL}/generate", data=form_data)

                    if response.status_code == 200:
                        response_text = response.text
                        try:
                            response_data = json.loads(response_text)
                            st.write(response_data.get("response"))
                            content_to_save = response_data
                        except json.JSONDecodeError:
                            st.write(response_text)
                            content_to_save = response_text

                        st.session_state.messages.append(
                            {"role": "assistant", "content": content_to_save}
                        )
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Failed to connect to the server: {str(e)}")

    with st.sidebar:
        st.title("About")
        st.markdown(
            """
        This is a RAG (Retrieval-Augmented Generation) chatbot that can answer
        questions based on the ingested PubMed documents in the field of Alzheimer's disease.
        """
        )

        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()


if __name__ == "__main__":
    main()
