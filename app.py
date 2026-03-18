import json
import re
import time
import unicodedata
from collections import Counter
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
MEMORY_FILE = Path("memory.json")
STREAM_RENDER_DELAY_SECONDS = 0.01
MEMORY_EXTRACTION_MAX_TOKENS = 200
ALLOWED_MEMORY_KEYS = {
    "name",
    "message_language",
    "preferred_language",
    "interests",
    "communication_style",
    "conversational_voice",
    "response_style",
    "writing_style",
    "favorite_topics",
}
LIST_STYLE_MEMORY_KEYS = {"interests", "favorite_topics"}
INFERABLE_STYLE_KEYS = {
    "communication_style",
    "conversational_voice",
    "response_style",
    "writing_style",
}
MEMORY_KEY_ALIASES = {
    "names": "name",
    "message_language": "message_language",
    "detected_language": "message_language",
    "interest": "interests",
    "hobby": "interests",
    "hobbies": "interests",
    "user_interests": "interests",
    "fav_topics": "favorite_topics",
    "topic": "favorite_topics",
    "topics": "favorite_topics",
    "favorite_topic": "favorite_topics",
    "favourite_topics": "favorite_topics",
    "favourite_topic": "favorite_topics",
    "preferred_topics": "favorite_topics",
    "language": "preferred_language",
    "primary_language": "preferred_language",
    "language_type": "preferred_language",
    "preferredlanguage": "preferred_language",
    "communication": "communication_style",
    "style": "communication_style",
    "voice": "conversational_voice",
    "conversation_voice": "conversational_voice",
    "conversational_voice": "conversational_voice",
    "response_style": "response_style",
    "writing_style": "writing_style",
    "response_tone": "response_style",
    "writing_tone": "writing_style",
    "tone": "communication_style",
    "communication_preference": "communication_style",
}
MEMORY_EXTRACTION_PROMPT = (
    "Return ONLY a valid JSON object with zero or more of these keys: "
    "name, message_language, preferred_language, interests, communication_style, "
    "conversational_voice, response_style, writing_style, favorite_topics. "
    "message_language is the language of the user's latest message. "
    "preferred_language is the language the user wants the assistant to use. "
    "Set preferred_language only if explicitly requested, or if very strong repeated "
    "evidence exists across messages. Do not infer preferred_language from one message "
    "unless the user clearly requests that language. "
    "For name and interests, include only explicitly stated traits. "
    "For communication_style, conversational_voice, and response_style, "
    "inference is allowed from a single message only when confidence is high. "
    "If uncertain, omit the field. Do not generate unsupported facts. "
    "Use JSON arrays for name, interests, and favorite_topics. "
    "Use short strings for preferred_language, communication_style, conversational_voice, "
    "response_style, and writing_style. "
    "Copy trait phrases from the user's message as closely as possible. "
    "If nothing explicit is stated, return {}."
)


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


def build_personalization_system_prompt(user_memory: dict[str, object]) -> str:
    memory_json = json.dumps(user_memory, ensure_ascii=True)
    return (
        "You are a helpful assistant. Personalize your responses towards these user "
        "traits whenever relevant. Use only the provided memory and do not invent "
        f"new traits. User memory JSON: {memory_json}"
    )


def stream_hf_router(messages: list[dict[str, str]], user_memory: dict[str, object]):
    hf_token = get_hf_token()

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    api_messages = [
        {"role": "system", "content": build_personalization_system_prompt(user_memory)},
        *messages,
    ]
    payload = {
        "model": HF_MODEL,
        "messages": api_messages,
        "max_tokens": MAX_TOKENS,
        "stream": True,
    }

    try:
        response = requests.post(
            HF_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
            stream=True,
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

    try:
        if response.status_code >= 400:
            details = response.text.strip()
            if len(details) > 300:
                details = f"{details[:300]}..."
            raise RuntimeError(
                f"Hugging Face API error ({response.status_code}). "
                f"{details or 'No additional details provided.'}"
            )

        saw_text_chunk = False
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if not line.startswith("data:"):
                continue

            data_str = line[len("data:") :].strip()
            if not data_str:
                continue
            if data_str == "[DONE]":
                break

            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            token_text = ""
            choices = event.get("choices")
            if isinstance(choices, list) and choices:
                first_choice = choices[0]
                if isinstance(first_choice, dict):
                    delta = first_choice.get("delta")
                    if isinstance(delta, dict):
                        content = delta.get("content")
                        if isinstance(content, str):
                            token_text = content
                    if not token_text:
                        message = first_choice.get("message")
                        if isinstance(message, dict):
                            content = message.get("content")
                            if isinstance(content, str):
                                token_text = content

            if token_text:
                saw_text_chunk = True
                time.sleep(STREAM_RENDER_DELAY_SECONDS)
                yield token_text

        if not saw_text_chunk:
            raise RuntimeError(
                "Received an unexpected streaming response format from Hugging Face."
            )
    finally:
        response.close()


def parse_json_object(text: str) -> dict[str, object]:
    stripped = text.strip()
    if not stripped:
        return {}

    try:
        data = json.loads(stripped)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
        try:
            data = json.loads(stripped)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = stripped[start : end + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            return {}

    return {}


def parse_key_value_traits(text: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("-* ").strip()
        if not line or ":" not in line:
            continue
        key_part, value_part = line.split(":", 1)
        key = key_part.strip().lower().replace(" ", "_")
        value = value_part.strip()
        if key and value:
            parsed[key] = value
    return parsed


def dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduped.append(value)
    return deduped


def normalize_names(value: object) -> list[str]:
    raw_names: list[str] = []

    if isinstance(value, str):
        for line in value.splitlines():
            for chunk in line.split(","):
                cleaned = re.sub(r"^\s*\d+[\).\-\s]*", "", chunk).strip()
                if cleaned:
                    raw_names.append(cleaned)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                for normalized in normalize_names(item):
                    raw_names.append(normalized)

    return dedupe_keep_order(raw_names)


def normalize_list_trait(value: object) -> list[str]:
    items: list[str] = []

    if isinstance(value, str):
        for line in value.splitlines():
            for chunk in re.split(r"[,;]", line):
                cleaned = re.sub(r"^\s*\d+[\).\-\s]*", "", chunk).strip()
                cleaned = re.sub(
                    (
                        r"^(?:(?:i|i'm|im|he|she|they|we|you)\s+)?"
                        r"(?:(?:also|really|mostly|often|usually)\s+)?"
                        r"(?:like|likes|love|loves|enjoy|enjoys|prefer|prefers|"
                        r"am\s+into|is\s+into|are\s+into|interested\s+in)\s+"
                    ),
                    "",
                    cleaned,
                    flags=re.IGNORECASE,
                )
                if cleaned:
                    items.append(cleaned)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, str):
                items.extend(normalize_list_trait(item))
    elif isinstance(value, dict):
        for dict_key, dict_value in value.items():
            if isinstance(dict_key, str):
                items.extend(normalize_list_trait(dict_key))
            if isinstance(dict_value, str):
                items.extend(normalize_list_trait(dict_value))
            elif isinstance(dict_value, list):
                items.extend(normalize_list_trait(dict_value))

    return dedupe_list_trait_semantic(items)


def normalize_trait_item_for_display(value: str) -> str:
    cleaned = normalize_text_quotes(value).strip(" .!?:\"'`()[]{}")
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        (
            r"^(?:my\s+hobb(?:y|ies)\s+(?:is|are|include)\s+|"
            r"my\s+favorite\s+(?:hobby|hobbies|games?|subjects?|topics?)\s+"
            r"(?:is|are|include)\s+)"
        ),
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        (
            r"^(?:(?:i|i'm|im|he|she|they|we|you)\s+)?"
            r"(?:(?:am|is|are)\s+)?"
            r"(?:into|interested\s+in|a\s+fan\s+of)\s+"
        ),
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        (
            r"^(?:(?:really|mostly|often|usually|just)\s+)?"
            r"(?:play(?:ing)?|watch(?:ing)?|read(?:ing)?|study(?:ing)?|"
            r"learn(?:ing)?|do(?:ing)?|practice(?:ing)?|practise(?:ing)?)\s+"
        ),
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip(" .!?:\"'`()[]{}")


def canonical_trait_item_key(value: str) -> str:
    canonical = normalize_trait_item_for_display(value).lower()
    canonical = re.sub(r"[^a-z0-9\s]+", " ", canonical)
    return " ".join(canonical.split())


def dedupe_list_trait_semantic(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        display_value = normalize_trait_item_for_display(value)
        key = canonical_trait_item_key(display_value)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(display_value)
    return deduped


def normalize_text_quotes(text: str) -> str:
    return (
        text.replace("\u2019", "'")
        .replace("\u2018", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


LANGUAGE_CANONICAL_MAP = {
    "english": "English",
    "inglés": "English",
    "ingles": "English",
    "spanish": "Spanish",
    "español": "Spanish",
    "espanol": "Spanish",
    "castellano": "Spanish",
    "french": "French",
    "français": "French",
    "francais": "French",
    "german": "German",
    "deutsch": "German",
    "portuguese": "Portuguese",
    "português": "Portuguese",
    "portugues": "Portuguese",
    "italian": "Italian",
    "italiano": "Italian",
    "japanese": "Japanese",
    "nihongo": "Japanese",
    "korean": "Korean",
    "hangul": "Korean",
    "chinese": "Chinese",
    "mandarin": "Chinese",
    "mandarin chinese": "Chinese",
    "arabic": "Arabic",
    "hindi": "Hindi",
    "urdu": "Urdu",
    "russian": "Russian",
}
LANGUAGE_NAME_PATTERN = "|".join(
    sorted((re.escape(key) for key in LANGUAGE_CANONICAL_MAP.keys()), key=len, reverse=True)
)


def normalize_language_name(language_text: str) -> str | None:
    candidate = normalize_text_quotes(language_text).strip().lower()
    if not candidate:
        return None
    compact = re.sub(r"[^a-z\u00c0-\u024f\s]+", " ", candidate)
    compact = " ".join(compact.split())
    if not compact:
        return None

    if compact in LANGUAGE_CANONICAL_MAP:
        return LANGUAGE_CANONICAL_MAP[compact]

    compact_ascii = strip_accents(compact)
    for raw_name, canonical_name in LANGUAGE_CANONICAL_MAP.items():
        if strip_accents(raw_name) == compact_ascii:
            return canonical_name
    return None


def detect_message_language(user_message: str) -> str | None:
    text = normalize_text_quotes(user_message)
    if not text.strip():
        return None

    if re.search(r"[\u3040-\u30FF]", text):
        return "Japanese"
    if re.search(r"[\uAC00-\uD7A3]", text):
        return "Korean"
    if re.search(r"[\u4E00-\u9FFF]", text):
        return "Chinese"
    if re.search(r"[\u0600-\u06FF]", text):
        return "Arabic"
    if re.search(r"[\u0400-\u04FF]", text):
        return "Russian"

    tokens = re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())
    if not tokens:
        return None

    language_markers = {
        "English": {
            "the",
            "and",
            "you",
            "please",
            "thanks",
            "hello",
            "my",
            "is",
            "are",
        },
        "Spanish": {
            "hola",
            "gracias",
            "por",
            "para",
            "quiero",
            "prefiero",
            "es",
            "que",
            "el",
            "la",
        },
        "French": {
            "bonjour",
            "merci",
            "je",
            "vous",
            "avec",
            "pour",
            "est",
            "le",
            "la",
        },
        "Portuguese": {
            "olá",
            "ola",
            "obrigado",
            "você",
            "voce",
            "para",
            "com",
            "que",
            "é",
            "de",
        },
        "German": {
            "hallo",
            "danke",
            "bitte",
            "ich",
            "und",
            "mit",
            "für",
            "fuer",
            "ist",
            "die",
        },
        "Italian": {
            "ciao",
            "grazie",
            "per",
            "con",
            "che",
            "sono",
            "il",
            "la",
        },
    }

    scores: dict[str, int] = {}
    token_set = set(tokens)
    for language_name, markers in language_markers.items():
        scores[language_name] = sum(1 for marker in markers if marker in token_set)

    best_language, best_score = max(scores.items(), key=lambda item: item[1])
    second_score = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0
    if best_score >= 2 and best_score > second_score:
        return best_language

    return None


def extract_explicit_preferred_language(user_message: str) -> str | None:
    text = normalize_text_quotes(user_message)
    patterns = [
        (
            r"\b(?:my\s+preferred\s+language\s+is|i\s+prefer|i(?:\s+would|'d)\s+prefer)\s+"
            r"(?:(?:you\s+)?(?:to\s+)?(?:speak|write|respond|reply|answer)\s+"
            r"(?:in\s+)?)?(?P<lang>"
            + LANGUAGE_NAME_PATTERN
            + r")\b"
        ),
        (
            r"\b(?:please\s+)?(?:speak|write|respond|reply|answer|talk)\s+"
            r"(?:(?:to\s+me\s+)?in\s+)?(?P<lang>"
            + LANGUAGE_NAME_PATTERN
            + r")\b"
        ),
        (
            r"\b(?:can|could)\s+you\s+(?:please\s+)?(?:speak|write|respond|reply|answer)\s+"
            r"(?:in\s+)?(?P<lang>"
            + LANGUAGE_NAME_PATTERN
            + r")\b"
        ),
        (
            r"\b(?:use|in)\s+(?P<lang>"
            + LANGUAGE_NAME_PATTERN
            + r")\s+(?:please)\b"
        ),
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        language_name = normalize_language_name(match.group("lang"))
        if language_name:
            return language_name
    return None


def infer_preferred_language_from_history(
    conversation_messages: list[dict[str, str]] | None,
) -> str | None:
    if not conversation_messages:
        return None

    detected_languages: list[str] = []
    for message in conversation_messages:
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        detected_language = detect_message_language(content)
        if detected_language:
            detected_languages.append(detected_language)

    if len(detected_languages) < 3:
        return None

    recent_languages = detected_languages[-5:]
    counts = Counter(recent_languages)
    top_language, top_count = counts.most_common(1)[0]
    if top_count >= 4:
        return top_language
    if len(recent_languages) == 3 and top_count == 3:
        return top_language
    return None


def phrase_in_text(phrase: str, text: str) -> bool:
    phrase_norm = re.sub(
        r"[^a-z0-9\s]+", " ", normalize_text_quotes(phrase).strip().lower()
    )
    text_norm = re.sub(r"[^a-z0-9\s]+", " ", normalize_text_quotes(text).strip().lower())
    phrase_norm = " ".join(phrase_norm.split())
    text_norm = " ".join(text_norm.split())
    if not phrase_norm:
        return False
    return phrase_norm in text_norm


def phrase_tokens_in_text_any_order(phrase: str, text: str) -> bool:
    phrase_tokens = re.findall(r"[a-z0-9]+", phrase.lower())
    text_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    ignored_tokens = {
        "and",
        "or",
        "the",
        "a",
        "an",
        "to",
        "in",
        "with",
        "please",
        "tone",
        "style",
        "voice",
        "responses",
        "response",
        "reply",
        "replies",
    }
    required_tokens = [token for token in phrase_tokens if token not in ignored_tokens]
    if not required_tokens:
        return False
    return all(token in text_tokens for token in required_tokens)


def normalize_scalar_trait(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def infer_response_style_from_message(user_message: str) -> str | None:
    lowered = user_message.lower()
    if re.search(r"\b(?:bullet|bullets|bullet points?)\b", lowered):
        return "Bullet points"
    if "step by step" in lowered:
        return "Step-by-step"
    if re.search(r"\b(?:short|brief|concise)\b", lowered):
        return "Concise"
    if re.search(r"\b(?:detailed|in-depth|in depth)\b", lowered):
        return "Detailed"
    return None


def infer_style_traits_from_message(user_message: str) -> dict[str, str]:
    inferred: dict[str, str] = {}
    stripped = user_message.strip()
    if not stripped:
        return inferred

    words = re.findall(r"[A-Za-z0-9']+", stripped)
    word_count = len(words)
    sentence_count = len([segment for segment in re.split(r"[.!?]+", stripped) if segment.strip()])
    has_complex_punct = any(char in stripped for char in [";", ":", "(", ")", "[", "]"])

    # Brief, plain prompts are high-confidence "Direct" communication style.
    if 2 <= word_count <= 14 and sentence_count <= 2 and not has_complex_punct:
        inferred["communication_style"] = "Direct"

    casual_markers = {
        "lol",
        "lmao",
        "idk",
        "btw",
        "pls",
        "thx",
        "gonna",
        "wanna",
        "kinda",
        "nah",
        "yep",
        "bro",
        "dude",
    }
    lowered_words = [token.lower() for token in words]
    alpha_chars = [char for char in stripped if char.isalpha()]
    lowercase_ratio = (
        sum(1 for char in alpha_chars if char.islower()) / len(alpha_chars)
        if alpha_chars
        else 0.0
    )
    has_casual_marker = any(token in casual_markers for token in lowered_words)
    if has_casual_marker or lowercase_ratio >= 0.95:
        inferred["conversational_voice"] = "Casual"

    response_style = infer_response_style_from_message(stripped)
    if response_style:
        inferred["response_style"] = response_style

    return inferred


def split_trait_items(raw_text: str) -> list[str]:
    items: list[str] = []
    chunks = re.split(
        r"\s*(?:,|;|/|&|\band\b|\bor\b)\s*",
        normalize_text_quotes(raw_text),
        flags=re.IGNORECASE,
    )
    ignore_values = {
        "",
        "it",
        "that",
        "this",
        "them",
        "things",
        "stuff",
        "nothing",
        "anything",
    }
    for chunk in chunks:
        cleaned = chunk.strip(" .!?:\"'`()[]{}")
        cleaned = re.sub(r"^(?:to\s+)?(?:do|be|have)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
        cleaned = normalize_trait_item_for_display(cleaned)
        if not cleaned:
            continue
        if cleaned.lower() in ignore_values:
            continue
        items.append(cleaned)
    return dedupe_list_trait_semantic(items)


def extract_explicit_interests_from_message(user_message: str) -> list[str]:
    user_message = normalize_text_quotes(user_message)
    subject_pattern = r"(?:i|i'm|im|he|she|they|we|you)"
    adverb_pattern = r"(?:(?:also|really|mostly|often|usually|kind\s+of|kinda)\s+)?"
    patterns = [
        (
            rf"\b{subject_pattern}\s+{adverb_pattern}"
            r"(?:like|likes|love|loves|enjoy|enjoys|prefer|prefers)\s+([^.!?\n]+)"
        ),
        rf"\b{subject_pattern}\s+(?:am|is|are)?\s*into\s+([^.!?\n]+)",
        rf"\b{subject_pattern}\s+(?:am|is|are)\s+interested\s+in\s+([^.!?\n]+)",
        (
            rf"\b{subject_pattern}\s+{adverb_pattern}"
            r"(?:play|plays|playing|watch|watches|watching|read|reads|reading|"
            r"listen\s+to|listens\s+to|listening\s+to|do|does|doing|practice|"
            r"practices|practicing)\s+([^.!?\n]+)"
        ),
        rf"\b{subject_pattern}\s+(?:am|is|are)\s+a\s+fan\s+of\s+([^.!?\n]+)",
        rf"\b(?:i|i'm|im)\s+follow\s+([^.!?\n]+)",
        r"\bmy\s+hobb(?:y|ies)\s+(?:is|are|include)\s+([^.!?\n]+)",
        r"\bmy\s+interests?\s+(?:are|include)\s+([^.!?\n]+)",
        r"\bmy\s+favorite\s+(?:hobby|hobbies|games?|subjects?)\s+(?:is|are)\s+([^.!?\n]+)",
        r"\bmy\s+favorite\s+topics?\s+(?:are|include)\s+([^.!?\n]+)",
    ]

    extracted_items: list[str] = []
    for pattern in patterns:
        for match in re.finditer(pattern, user_message, flags=re.IGNORECASE):
            extracted_items.extend(split_trait_items(match.group(1)))

    return dedupe_keep_order(extracted_items)


def filter_explicit_traits(
    extracted: dict[str, object], source_user_message: str
) -> dict[str, object]:
    filtered: dict[str, object] = {}

    for key, value in extracted.items():
        normalized_key = MEMORY_KEY_ALIASES.get(key, key)
        if normalized_key not in ALLOWED_MEMORY_KEYS:
            continue

        if normalized_key == "name":
            explicit_names = [
                name
                for name in normalize_names(value)
                if phrase_in_text(name, source_user_message)
            ]
            if explicit_names:
                filtered["name"] = dedupe_keep_order(explicit_names)
            continue

        if normalized_key in {"message_language", "preferred_language"}:
            # Language handling is deterministic in extract_user_traits.
            continue

        if normalized_key in LIST_STYLE_MEMORY_KEYS:
            explicit_items = [
                item
                for item in normalize_list_trait(value)
                if phrase_in_text(item, source_user_message)
            ]
            if explicit_items:
                filtered[normalized_key] = dedupe_list_trait_semantic(explicit_items)
            continue

        if normalized_key in INFERABLE_STYLE_KEYS:
            style_value = normalize_scalar_trait(value)
            if style_value and (
                phrase_in_text(style_value, source_user_message)
                or phrase_tokens_in_text_any_order(style_value, source_user_message)
            ):
                filtered[normalized_key] = style_value
            continue

        if isinstance(value, str):
            clean_value = value.strip()
            if clean_value and (
                phrase_in_text(clean_value, source_user_message)
                or phrase_tokens_in_text_any_order(clean_value, source_user_message)
            ):
                filtered[normalized_key] = clean_value
            continue

        if isinstance(value, list):
            clean_items = []
            for item in value:
                if not isinstance(item, str):
                    continue
                clean_item = item.strip()
                if clean_item and (
                    phrase_in_text(clean_item, source_user_message)
                    or phrase_tokens_in_text_any_order(clean_item, source_user_message)
                ):
                    clean_items.append(clean_item)
            if clean_items:
                filtered[normalized_key] = dedupe_keep_order(clean_items)

    return filtered


def extract_user_traits(
    user_message: str,
    conversation_messages: list[dict[str, str]] | None = None,
    existing_preferred_language: object = None,
) -> dict[str, object]:
    hf_token = get_hf_token()
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": HF_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only a valid JSON object with extracted traits. "
                    f"{MEMORY_EXTRACTION_PROMPT}"
                ),
            },
            {"role": "user", "content": user_message},
        ],
        "max_tokens": MEMORY_EXTRACTION_MAX_TOKENS,
    }

    try:
        response = requests.post(
            HF_CHAT_COMPLETIONS_URL,
            headers=headers,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("Memory extraction timed out.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError("Memory extraction request failed.") from exc

    if response.status_code >= 400:
        details = response.text.strip()
        if len(details) > 300:
            details = f"{details[:300]}..."
        raise RuntimeError(
            f"Memory extraction API error ({response.status_code}). "
            f"{details or 'No additional details provided.'}"
        )

    try:
        data = response.json()
        extracted_text = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("Unexpected memory extraction response format.") from exc

    if not isinstance(extracted_text, str):
        return {}
    extracted = parse_json_object(extracted_text)
    if not extracted:
        extracted = parse_key_value_traits(extracted_text)
    if not extracted:
        fallback_names = normalize_names(extracted_text)
        if fallback_names:
            extracted = {"name": fallback_names}

    filtered = filter_explicit_traits(extracted, user_message)
    direct_interests = extract_explicit_interests_from_message(user_message)
    if direct_interests:
        existing_interests = normalize_list_trait(filtered.get("interests", []))
        filtered["interests"] = dedupe_list_trait_semantic(existing_interests + direct_interests)

    filtered["message_language"] = detect_message_language(user_message)

    explicit_preferred_language = extract_explicit_preferred_language(user_message)
    if explicit_preferred_language:
        filtered["preferred_language"] = explicit_preferred_language
    elif not (
        isinstance(existing_preferred_language, str)
        and existing_preferred_language.strip()
    ):
        repeated_preferred_language = infer_preferred_language_from_history(
            conversation_messages
        )
        if repeated_preferred_language:
            filtered["preferred_language"] = repeated_preferred_language

    inferred_style_traits = infer_style_traits_from_message(user_message)
    for key, value in inferred_style_traits.items():
        filtered.setdefault(key, value)

    return filtered


def load_user_memory() -> dict[str, object]:
    if not MEMORY_FILE.exists():
        return {}

    try:
        with MEMORY_FILE.open("r", encoding="utf-8") as f:
            memory = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}

    if not isinstance(memory, dict):
        return {}

    normalized_memory: dict[str, object] = {}
    for key, value in memory.items():
        normalized_key = MEMORY_KEY_ALIASES.get(key, key)
        if normalized_key not in ALLOWED_MEMORY_KEYS:
            continue

        if normalized_key == "name":
            names = normalize_names(value)
            if names:
                normalized_memory["name"] = names
            continue

        if normalized_key in LIST_STYLE_MEMORY_KEYS:
            items = normalize_list_trait(value)
            if items:
                normalized_memory[normalized_key] = items
            continue

        if isinstance(value, str):
            clean_value = value.strip()
            if clean_value:
                normalized_memory[normalized_key] = clean_value
            continue

        if isinstance(value, list):
            clean_items = [item.strip() for item in value if isinstance(item, str) and item.strip()]
            if clean_items:
                normalized_memory[normalized_key] = dedupe_keep_order(clean_items)

    return normalized_memory


def save_user_memory(memory: dict[str, object]) -> None:
    with MEMORY_FILE.open("w", encoding="utf-8") as f:
        json.dump(memory, f, ensure_ascii=True, indent=2)


def merge_user_memory(
    existing_memory: dict[str, object], extracted_memory: dict[str, object]
) -> dict[str, object]:
    merged = dict(existing_memory)
    for key, value in extracted_memory.items():
        if key == "name":
            combined_names = normalize_names(merged.get("name", [])) + normalize_names(value)
            deduped_names = dedupe_keep_order(combined_names)
            if deduped_names:
                merged["name"] = deduped_names
            continue

        if key in LIST_STYLE_MEMORY_KEYS:
            combined_items = normalize_list_trait(merged.get(key, [])) + normalize_list_trait(
                value
            )
            deduped_items = dedupe_list_trait_semantic(combined_items)
            if deduped_items:
                merged[key] = deduped_items
            continue
        merged[key] = value
    return merged


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


def initialize_memory_state() -> None:
    if "user_memory" not in st.session_state:
        st.session_state["user_memory"] = load_user_memory()
    if "last_memory_error" not in st.session_state:
        st.session_state["last_memory_error"] = None


def shorten_title(text: str, max_length: int = 30) -> str:
    trimmed = text.strip()
    if len(trimmed) <= max_length:
        return trimmed
    return f"{trimmed[:max_length]}..."


st.set_page_config(page_title=APP_TITLE, layout="wide")
initialize_chat_state()
initialize_memory_state()

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

with st.sidebar.expander("User Memory", expanded=False):
    if st.session_state["user_memory"]:
        names = st.session_state["user_memory"].get("name", [])
        if isinstance(names, list) and names:
            st.markdown("**name**")
            for name in names:
                st.text(name)

        other_memory = {
            key: value
            for key, value in st.session_state["user_memory"].items()
            if key != "name"
        }
        if other_memory:
            st.json(other_memory)
    else:
        st.caption("No memory saved yet.")
    if st.session_state.get("last_memory_error"):
        st.warning(f"Memory extraction issue: {st.session_state['last_memory_error']}")
    if st.button("Clear Memory", use_container_width=True):
        st.session_state["user_memory"] = {}
        st.session_state["last_memory_error"] = None
        save_user_memory(st.session_state["user_memory"])
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

    with st.chat_message("assistant"):
        try:
            assistant_text = st.write_stream(
                stream_hf_router(active_chat["messages"], st.session_state["user_memory"])
            )
        except RuntimeError as exc:
            st.error(str(exc))
        else:
            active_chat["messages"].append({"role": "assistant", "content": assistant_text})
            save_chat(active_chat)

            try:
                extracted_memory = extract_user_traits(
                    user_prompt,
                    active_chat["messages"],
                    st.session_state["user_memory"].get("preferred_language"),
                )
            except RuntimeError as exc:
                extracted_memory = {}
                st.session_state["last_memory_error"] = str(exc)
            if extracted_memory:
                st.session_state["user_memory"] = merge_user_memory(
                    st.session_state["user_memory"], extracted_memory
                )
                save_user_memory(st.session_state["user_memory"])
                st.session_state["last_memory_error"] = None

            st.rerun()
