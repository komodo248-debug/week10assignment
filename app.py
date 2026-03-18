import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import requests
import streamlit as st


HF_CHAT_COMPLETIONS_URL = "https://router.huggingface.co/v1/chat/completions"
HF_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
REQUEST_TIMEOUT_SECONDS = 20
MAX_TOKENS = 512
APP_TITLE = "My AI Chat"
DEFAULT_CHAT_TITLE = "New Chat"
CHATS_DIR = Path("chats")


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


def new_chat() -> dict[str, object]:
    return {
        "id": str(uuid4()),
        "title": DEFAULT_CHAT_TITLE,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": [],
    }


def chat_file_path(chat_id: str) -> Path:
    return CHATS_DIR / f"{chat_id}.json"


def is_valid_chat(chat: object) -> bool:
    if not isinstance(chat, dict):
        return False

    chat_id = chat.get("id")
    title = chat.get("title")
    created_at = chat.get("created_at")
    messages = chat.get("messages")
    if not isinstance(chat_id, str) or not chat_id.strip():
        return False
    if not isinstance(title, str):
        return False
    if not isinstance(created_at, str):
        return False
    if not isinstance(messages, list):
        return False

    for message in messages:
        if not isinstance(message, dict):
            return False
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            return False
    return True


def save_chat(chat: dict[str, object]) -> None:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    with chat_file_path(chat["id"]).open("w", encoding="utf-8") as f:
        json.dump(chat, f, ensure_ascii=True, indent=2)


def load_chats() -> tuple[list[dict[str, object]], int]:
    CHATS_DIR.mkdir(parents=True, exist_ok=True)
    loaded_chats: list[dict[str, object]] = []
    skipped = 0
    seen_ids: set[str] = set()

    for path in sorted(CHATS_DIR.glob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as f:
                chat = json.load(f)
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue

        if not is_valid_chat(chat):
            skipped += 1
            continue
        if chat["id"] in seen_ids:
            skipped += 1
            continue

        seen_ids.add(chat["id"])
        loaded_chats.append(chat)

    return loaded_chats, skipped


def delete_chat_file(chat_id: str) -> None:
    try:
        chat_file_path(chat_id).unlink(missing_ok=True)
    except OSError:
        pass


def get_active_chat() -> dict[str, object] | None:
    active_chat_id = st.session_state.get("active_chat_id")
    for chat in st.session_state.get("chats", []):
        if chat["id"] == active_chat_id:
            return chat
    return None


def initialize_chat_state() -> None:
    if "chats" not in st.session_state:
        loaded_chats, skipped = load_chats()
        st.session_state["chats"] = loaded_chats
        st.session_state["active_chat_id"] = loaded_chats[0]["id"] if loaded_chats else None
        st.session_state["skipped_chat_files"] = skipped
        return

    if "active_chat_id" not in st.session_state:
        chats = st.session_state["chats"]
        st.session_state["active_chat_id"] = chats[0]["id"] if chats else None
        return

    if "skipped_chat_files" not in st.session_state:
        st.session_state["skipped_chat_files"] = 0

    if get_active_chat() is None:
        chats = st.session_state["chats"]
        st.session_state["active_chat_id"] = chats[0]["id"] if chats else None


def shorten_title(text: str, max_length: int = 30) -> str:
    trimmed = text.strip()
    if len(trimmed) <= max_length:
        return trimmed
    return f"{trimmed[:max_length]}..."


st.set_page_config(page_title=APP_TITLE, layout="wide")
initialize_chat_state()

st.sidebar.header("Chats")
if st.session_state.get("skipped_chat_files", 0) > 0:
    st.sidebar.warning(
        f"Skipped {st.session_state['skipped_chat_files']} invalid chat file(s)."
    )

if st.sidebar.button("New Chat", use_container_width=True):
    created_chat = new_chat()
    st.session_state["chats"].append(created_chat)
    st.session_state["active_chat_id"] = created_chat["id"]
    save_chat(created_chat)
    st.rerun()

switch_to_id = None
delete_chat_id = None

chat_list_container = st.sidebar.container(height=500)
with chat_list_container:
    if not st.session_state["chats"]:
        st.info("No chats yet.")
    for chat in st.session_state["chats"]:
        with st.container(border=True):
            is_active = chat["id"] == st.session_state["active_chat_id"]
            row_cols = st.columns([5, 1])
            if row_cols[0].button(
                chat["title"],
                key=f"chat_switch_{chat['id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                switch_to_id = chat["id"]
            if row_cols[1].button(
                "✕",
                key=f"chat_delete_{chat['id']}",
                use_container_width=True,
                type="secondary",
            ):
                delete_chat_id = chat["id"]
            st.caption(chat["created_at"])

if delete_chat_id is not None:
    remaining_chats = [
        chat for chat in st.session_state["chats"] if chat["id"] != delete_chat_id
    ]
    st.session_state["chats"] = remaining_chats
    delete_chat_file(delete_chat_id)
    if st.session_state["active_chat_id"] == delete_chat_id:
        st.session_state["active_chat_id"] = (
            remaining_chats[0]["id"] if remaining_chats else None
        )
    st.rerun()

if switch_to_id is not None and switch_to_id != st.session_state["active_chat_id"]:
    st.session_state["active_chat_id"] = switch_to_id
    st.rerun()

st.title(APP_TITLE)

history_container = st.container(height=500)
active_chat = get_active_chat()

with history_container:
    if active_chat is None:
        st.info("No active chat. Create a new chat from the sidebar.")
    elif not active_chat["messages"]:
        st.info("Start the conversation by sending a message below.")
    elif active_chat is not None:
        for message in active_chat["messages"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

user_prompt = st.chat_input("Type your message", disabled=active_chat is None)
if user_prompt and active_chat is not None:
    active_chat["messages"].append({"role": "user", "content": user_prompt})
    if active_chat["title"] == DEFAULT_CHAT_TITLE:
        active_chat["title"] = shorten_title(user_prompt)
    save_chat(active_chat)

    with st.spinner("Calling Hugging Face API..."):
        try:
            assistant_text = call_hf_router(active_chat["messages"])
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            active_chat["messages"].append({"role": "assistant", "content": assistant_text})
            save_chat(active_chat)
            st.rerun()
