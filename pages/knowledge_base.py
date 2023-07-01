import streamlit as st
from streamlit_tree_select import tree_select
from connections import fetch_namespaces

st.title("🗄️ Remote knowledge base")
st.subheader("Select your resources")

if "namespaces" not in st.session_state:
    st.session_state["namespaces"] = fetch_namespaces()


directory = []

for index, namespace in enumerate(st.session_state["namespaces"]):
    directory.append(
        {
            "label": namespace,
            "value": namespace,
        }
    )

return_select = tree_select(directory)
st.write(return_select)
