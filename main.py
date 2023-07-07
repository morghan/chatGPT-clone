import streamlit as st

from conversation_handlers import chat_completion_request, execute_function_call
from connections import fetch_system_prompt, upload_prompt
from streamlit_handlers import init, display_chat_history, submit


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

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {
            "role": "system",
            "content": st.session_state["functions_instructions"]
            + st.session_state["custom_prompt"],
        }
    ]

if "something" not in st.session_state:
    st.session_state["something"] = ""

if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "chat_reset" not in st.session_state:
    st.session_state["chat_reset"] = False

if "current_page" not in st.session_state:
    st.session_state["current_page"] = "main"


def main():
    with st.sidebar:
        st.write(st.session_state["messages"])
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
                    st.session_state["custom_prompt"] = file_content
                    st.session_state["messages"][0] = {
                        "role": "system",
                        "content": st.session_state["functions_instructions"]
                        + st.session_state["custom_prompt"],
                    }

                    st.session_state["chat_reset"] = True
                    if "chat_history" in st.session_state:
                        del st.session_state["chat_history"]

        st.text_input("Your input", key="user_input", on_change=submit)

    if uploaded_file is not None:
        st.success("Custom prompt uploaded successfully.", icon="📝")

    if (
        st.session_state["something"] != ""
        and not st.session_state["chat_reset"]
        and st.session_state["current_page"] == "main"
    ):
        with st.spinner("Thinking..."):
            st.session_state["messages"].append(
                {"role": "user", "content": st.session_state["something"]}
            )
            chat_response = chat_completion_request(
                st.session_state["messages"], functions=st.session_state["functions"]
            )
            assistant_message = chat_response.json()["choices"][0]["message"]
            st.session_state["messages"].append(assistant_message)
            if assistant_message.get("function_call"):
                results = execute_function_call(assistant_message)
                st.session_state["messages"].append(
                    {
                        "role": "function",
                        "name": assistant_message["function_call"]["name"],
                        "content": results,
                    }
                )

            st.session_state["chat_history"].append(
                (
                    st.session_state["something"],
                    results
                    if assistant_message.get("function_call")
                    else assistant_message["content"],
                )
            )

    display_chat_history()
    st.session_state["current_page"] = "main"


if __name__ == "__main__":
    init()
    main()
