import streamlit as st
import time

from conversation_handlers import chat_completion_request, execute_function_call
from connections import fetch_system_prompt, upload_prompt
from streamlit_handlers import (
    init,
    render_conversation,
    render_qa_agent,
)
from langchain.callbacks import StreamlitCallbackHandler


def build_custom_prompt_suffix():
    st.session_state[
        "functions_instructions"
    ] = """FUNCTIONS INSTRUCTIONS: Only use the functions you have been provided with.
    Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous.
    """


if "functions_instructions" not in st.session_state:
    build_custom_prompt_suffix()

if "custom_prompt" not in st.session_state:
    st.session_state["custom_prompt"] = fetch_system_prompt()

if "functions" not in st.session_state:
    st.session_state["functions"] = [
        {
            "name": "respond_franchise_inquiry",
            "description": "Responds to inquiries about franchises. The inquiry should be a fully formed question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "inquiry": {
                        "type": "string",
                        "description": "The inquiry about a or multiple franchises.",
                    },
                },
                "required": ["inquiry"],
            },
        },
    ]

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [
        {
            "role": "system",
            "content": st.session_state["functions_instructions"]
            + st.session_state["custom_prompt"],
        }
    ]


def reset_chat(custom_prompt):
    st.session_state["custom_prompt"] = custom_prompt
    st.session_state["chat_history"] = [
        {
            "role": "system",
            "content": st.session_state["functions_instructions"]
            + st.session_state["custom_prompt"],
        }
    ]


def main():
    with st.sidebar:
        # System Message upload
        with st.form("system_prompt", clear_on_submit=True):
            uploaded_file = st.file_uploader(
                "You can customize the system prompt by uploading a .txt file. This prompt will be saved for future sessions",
                type=["txt"],
            )
            submitted1 = st.form_submit_button("Submit")
            if submitted1 and uploaded_file is not None:
                with st.spinner("Uploading and processing prompt 🚧 ..."):
                    file_content = uploaded_file.getvalue().decode("utf-8").strip()
                    upload_prompt(file_content)
                    reset_chat(file_content)

        render_qa_agent()

    if uploaded_file is not None:
        st.success("Custom prompt uploaded successfully.", icon="📝")

    # Render conversation
    render_conversation()

    # Check for user input
    if prompt := st.chat_input("Your input"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state["chat_history"].append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            # Placeholders
            message_placeholder = st.empty()
            function_message = {
                "content": None,
                "function_call": {"name": None, "arguments": ""},
            }
            full_response = ""
            # Stream chat completion
            for chat_response in chat_completion_request(
                messages=st.session_state["chat_history"],
                functions=st.session_state["functions"],
            ):
                # Streamed chunk
                delta = chat_response["choices"][0]["delta"]

                # Checks if LLM is responding by itself
                if "content" in delta and "function_call" not in delta:
                    full_response += delta.get("content", "")
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                if chat_response["choices"][0]["finish_reason"] == "stop":
                    message_placeholder.markdown(full_response)
                    st.session_state["chat_history"].append(
                        {"role": "assistant", "content": full_response}
                    )

                # Checks if LLM needs to call a function to respond
                if "function_call" in delta:
                    if "name" in delta["function_call"]:
                        function_message["function_call"]["name"] = delta[
                            "function_call"
                        ]["name"]
                        function_message["content"] = delta["content"]
                    if "arguments" in delta["function_call"]:
                        function_message["function_call"]["arguments"] += delta[
                            "function_call"
                        ]["arguments"]
                if chat_response["choices"][0]["finish_reason"] == "function_call":
                    st.session_state["chat_history"].append(
                        {
                            "role": "assistant",
                            "content": function_message["content"],
                            "function_call": function_message["function_call"],
                        }
                    )
                    if st.session_state.get("agent") is not None:
                        results = execute_function_call(
                            function_message,
                            st_callback=StreamlitCallbackHandler(
                                st.container(),
                            ),
                        )
                        st.markdown(results)
                    else:
                        results = "I'm sorry, I don't have an QA Agent to respond to your inquiry."
                        function_response = ""
                        for word in results.split(sep=" "):
                            function_response += word + " "
                            time.sleep(0.05)
                            message_placeholder.markdown(function_response + "▌")
                        message_placeholder.markdown(function_response)

                    st.session_state["chat_history"].append(
                        {
                            "role": "function",
                            "name": function_message["function_call"]["name"],
                            "content": results,
                        }
                    )


if __name__ == "__main__":
    init()
    main()
