from htbuilder.units import rem
from htbuilder import div, styles
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import datetime
import textwrap
import time
import os
import pickle

import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="Windpark Lindenberg Assistant", page_icon="🌱")

# Configuration
MODEL = "gpt-4o-mini"
HISTORY_LENGTH = 5
SUMMARIZE_OLD_HISTORY = True
CONTEXT_LEN = 3
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)

DEBUG_MODE = st.query_params.get("debug", "false").lower() == "true"

INSTRUCTIONS = textwrap.dedent("""
    - Du bist ein hilfreicher AI-Chat-Assistent für das Windpark Lindenberg Projekt in Beinwil (Freiamt), Schweiz.
    - Du erhältst zusätzliche Informationen aus dem offiziellen Planungsbericht.
    - Verwende den Kontext und die Historie, um kohärente Antworten zu geben.
    - Sei präzise, aber klar. Schreibe 2-3 Sätze maximal.
    - Antworte NUR mit Informationen aus den bereitgestellten Dokumenten.
    - Gib keine Rechtsberatung oder persönliche Meinungen ab.
    - Zitiere immer die Quelle: "Quelle: Lindenberg_Planungsbericht"
""")

SUGGESTIONS = {
    ":green[:material/nature:] Umweltauswirkungen": (
        "Welche Umweltauswirkungen hat der Windpark Lindenberg?"
    ),
    ":blue[:material/groups:] Bürgerbeteiligung": (
        "Wie können Bürger am Planungsverfahren teilnehmen?"
    ),
    ":orange[:material/bolt:] Energieproduktion": (
        "Wie viel Energie wird der Windpark produzieren?"
    ),
    ":violet[:material/schedule:] Zeitplan": (
        "Wann ist die Umsetzung des Windparks geplant?"
    ),
    ":red[:material/location_on:] Standort": (
        "Wo genau wird der Windpark gebaut und warum dort?"
    ),
}

# Load knowledge base
@st.cache_data
def load_lindenberg_knowledge_base():
    """Load the processed PDF chunks"""
    # For now, return sample data - you'll upload the real data later
    return [
        {
            "content": "Windpark Lindenberg ist ein Projekt in Beinwil (Freiamt), Aargau. Das Projekt befindet sich in der Planungsphase.",
            "source": "Lindenberg_Planungsbericht",
            "category": "general"
        }
    ]

def improved_search(query, documents, max_results=3):
    """Search function with scoring"""
    if not documents:
        return []
        
    query_words = query.lower().split()
    scored_docs = []
    
    for doc in documents:
        content_lower = doc["content"].lower()
        score = 0
        
        for word in query_words:
            if word in content_lower:
                score += content_lower.count(word)
        
        if query.lower() in content_lower:
            score += 5
            
        if score > 0:
            scored_docs.append((score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:max_results]]

def build_prompt(**kwargs):
    """Builds a prompt string with the kwargs as HTML-like tags."""
    prompt = []
    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")
    return "\n".join(prompt)

def build_question_prompt(question):
    """Fetches info from documents and creates the prompt string."""
    documents = load_lindenberg_knowledge_base()
    
    old_history = st.session_state.messages[:-HISTORY_LENGTH] if len(st.session_state.messages) > HISTORY_LENGTH else []
    recent_history = st.session_state.messages[-HISTORY_LENGTH:] if st.session_state.messages else []

    recent_history_str = history_to_text(recent_history) if recent_history else None
    relevant_docs = improved_search(question, documents, max_results=CONTEXT_LEN)
    
    if relevant_docs:
        context_str = "\n\n".join([f"Quelle: {doc['source']}\n{doc['content']}" for doc in relevant_docs])
    else:
        context_str = "Keine relevanten Informationen in den Dokumenten gefunden."

    return build_prompt(
        instructions=INSTRUCTIONS,
        document_context=context_str,
        recent_messages=recent_history_str,
        question=question,
    )

def history_to_text(chat_history):
    """Converts chat history into a string."""
    return "\n".join(f"[{h['role']}]: {h['content']}" for h in chat_history)

def get_response(prompt):
    """Get streaming response from OpenAI"""
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.1,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
                
    except Exception as e:
        yield f"Fehler bei der Antwortgenerierung: {str(e)}"

@st.dialog("Rechtlicher Hinweis")
def show_disclaimer_dialog():
    st.caption("""
            Dieser AI-Chatbot stellt Informationen über das Windpark Lindenberg Projekt bereit.
            Die Antworten basieren ausschließlich auf dem offiziellen Planungsbericht.
            Antworten können unvollständig oder ungenau sein. Es wird keine Rechtsberatung gegeben.
        """)

# UI
st.html(div(style=styles(font_size=rem(5), line_height=1))["🌱"])

title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title("Windpark Lindenberg Assistant", anchor=False, width="stretch")

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)

user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)

user_first_interaction = user_just_asked_initial_question or user_just_clicked_suggestion
has_message_history = "messages" in st.session_state and len(st.session_state.messages) > 0

if not user_first_interaction and not has_message_history:
    st.session_state.messages = []

    with st.container():
        st.chat_input("Stellen Sie eine Frage...", key="initial_question")
        selected_suggestion = st.pills(
            label="Beispiele",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    st.button(
        "&nbsp;:small[:gray[:material/balance: Rechtlicher Hinweis]]",
        type="tertiary",
        on_click=show_disclaimer_dialog,
    )
    st.stop()

user_message = st.chat_input("Stellen Sie eine Nachfrage...")

if not user_message:
    if user_just_asked_initial_question:
        user_message = st.session_state.initial_question
    if user_just_clicked_suggestion:
        user_message = SUGGESTIONS[st.session_state.selected_suggestion]

with title_row:
    def clear_conversation():
        st.session_state.messages = []
        st.session_state.initial_question = None
        st.session_state.selected_suggestion = None

    st.button("Neustart", icon=":material/refresh:", on_click=clear_conversation)

if "prev_question_timestamp" not in st.session_state:
    st.session_state.prev_question_timestamp = datetime.datetime.fromtimestamp(0)

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.container()
        st.markdown(message["content"])

if user_message:
    user_message = user_message.replace("$", r"\$")

    with st.chat_message("user"):
        st.text(user_message)

    with st.chat_message("assistant"):
        with st.spinner("Warten..."):
            question_timestamp = datetime.datetime.now()
            time_diff = question_timestamp - st.session_state.prev_question_timestamp
            st.session_state.prev_question_timestamp = question_timestamp

            if time_diff < MIN_TIME_BETWEEN_REQUESTS:
                time.sleep(time_diff.seconds + time_diff.microseconds * 0.001)

            user_message = user_message.replace("'", "")

        if DEBUG_MODE:
            with st.status("Berechne Prompt...") as status:
                full_prompt = build_question_prompt(user_message)
                st.code(full_prompt)
                status.update(label="Prompt berechnet")
        else:
            with st.spinner("Recherchiere..."):
                full_prompt = build_question_prompt(user_message)

        with st.spinner("Denke nach..."):
            response_gen = get_response(full_prompt)

        with st.container():
            response = st.write_stream(response_gen)
            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})
