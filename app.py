import requests
import streamlit as st


HF_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
REQUEST_TIMEOUT_SECONDS = 20
MAX_TOKENS = 512


def get_hf_token() -> str:
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception as exc:
        raise RuntimeError(
            "Missing Hugging Face token. Add HF_TOKEN to .streamlit/secrets.toml "
            "or Streamlit Cloud Advanced settings."
        ) from exc
    if not isinstance(hf_token, str) or not hf_token.strip():
        raise RuntimeError(
            "HF_TOKEN is empty. Add a valid token in .streamlit/secrets.toml "
            "or Streamlit Cloud Advanced settings."
        )
    return hf_token.strip()


def call_hf_router(messages: list[dict[str, str]]) -> str:
    hf_token = get_hf_token()

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
    }

    try:
        response = requests.post(
            HF_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "The request timed out. Please try again in a moment."
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "Network error while contacting Hugging Face. Please check your "
            "connection and try again."
        ) from exc

    if response.status_code >= 400:
        details = response.text.strip()
        if len(details) > 300:
            details = f"{details[:300]}..."
        raise RuntimeError(
            f"Hugging Face API error ({response.status_code}). "
            f"{details or 'No additional details provided.'}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(
            "Received an invalid JSON response from Hugging Face."
        ) from exc

    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "Received an unexpected response format from Hugging Face."
        ) from exc


st.set_page_config(page_title="My AI Chat", layout="wide")
st.title("My AI Chat")
st.subheader("Part B: Multi-Turn Conversation UI")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

history_container = st.container(height=500)
with history_container:
    if not st.session_state["messages"]:
        st.info("Start the conversation by sending a message below.")
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

user_prompt = st.chat_input("Type your message")
if user_prompt:
    st.session_state["messages"].append({"role": "user", "content": user_prompt})
    with st.spinner("Calling Hugging Face API..."):
        try:
            assistant_text = call_hf_router(st.session_state["messages"])
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            st.session_state["messages"].append(
                {"role": "assistant", "content": assistant_text}
            )
            st.rerun()
