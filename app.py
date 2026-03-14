import streamlit as st
import requests
import json
import datetime
import textwrap
import time

st.set_page_config(page_title="Windpark Lindenberg Assistant (Apertus)", page_icon="🌱")

# Configuration
MODEL = "swiss-ai/apertus-8b-instruct"
HISTORY_LENGTH = 5
CONTEXT_LEN = 3
MIN_TIME_BETWEEN_REQUESTS = datetime.timedelta(seconds=1)

INSTRUCTIONS = textwrap.dedent("""
    ## Grundlegende Rolle und Zweck
    - Du bist ein neutraler Informationsassistent und Reflexionsbegleiter für das Windpark Lindenberg Projekt in Beinwil (Freiamt), Schweiz.
    - Dein Zweck ist es, Stakeholder über das Projekt zu informieren und zur Selbstreflexion anzuregen.
    - Du ersetzt keine demokratischen Partizipationsprozesse, sondern bereitest darauf vor.

    ## Antwortrichtlinien
    - **Nur aus der Wissensbasis antworten**: Verwende ausschließlich Informationen aus den hochgeladenen Projektdokumenten.
    - **Neutrale, klare Sprache**: Verwende kurze, verständliche und neutrale Formulierungen (2-3 Kernsätze).
    - **Explizite Behandlung von Trade-offs**: Stelle lokale vs. globale und kurzfristige vs. langfristige Abwägungen explizit dar.
    - **Unsicherheiten offenlegen**: Kommuniziere Unsicherheiten transparent.
    - **Quellen zitieren**: Füge 1-3 Quellenangaben am Ende jeder Antwort hinzu: "Quelle: Lindenberg_Planungsbericht".

    ## Psychologische Reflexionsbegleitung
    - **Alle Emotionen als legitim validieren**: Erkenne Gefühle wie Kontrollverlust, Bedrohung oder Ungerechtigkeit als berechtigt an.
    - **Emotionen in sozialen Kontext einbetten**: Helfe Nutzern dabei, persönliche Emotionen mit breiteren gesellschaftlichen Zusammenhängen zu verbinden.
    - **Reflexions-Framework anwenden**:
      1. **Wahrnehmen**: Frage nach konkreten Erfahrungen mit dem Projekt
      2. **Verstehen**: Erkunde Muster und Verbindungen zu vergangenen Erfahrungen
      3. **Zukunft gestalten**: Leite zu zukünftigen Handlungsmöglichkeiten an

    ## Perspektiven-Prompts (optional)
    - Biete kontextsensitive Reflexionshilfen an: "Möchten Sie das Projekt aus der Perspektive von Nachbarn/Ökosystem/zukünftigen Generationen betrachten?"
    - Erkunde räumliche (lokal vs. global) und zeitliche (kurzfristig vs. langfristig) Dimensionen.
    - Markiere diese deutlich als optionale Reflexionshilfen, nicht als Überzeugungselemente.

    ## Fallback-Verhalten
    - Wenn keine relevanten Informationen verfügbar sind, informiere transparent darüber.
    - Schlage verwandte Themen aus der Wissensbasis vor.
    - Verwende klaren Fallback-Text: "Zu dieser Frage finde ich keine direkten Informationen im Planungsbericht."

    ## Verbotene Aktivitäten
    - Keine Rechtsberatung oder persönliche Meinungen
    - Keine Meinungsaggregation oder Speicherung persönlicher Daten
    - Keine externen Informationen verwenden
    - Keine offizielle Positionen vertreten
    - Keine Manipulation oder Überzeugungsversuche

    ## Transparenz und Ethik
    - Kommuniziere deine Rolle als Informations- und Reflexionswerkzeug transparent.
    - Betone freiwillige und anonyme Nutzung.
    - Schaffe einen urteilsfreien Reflexionsraum ohne sozialen Druck.
    - Respektiere das eigene Tempo der Nutzer bei der Reflexion.
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

@st.cache_data
def load_lindenberg_knowledge_base():
    """Load sample knowledge base - replace with your real data later"""
    return [
        {
            "content": "Windpark Lindenberg ist ein Projekt in Beinwil (Freiamt), Aargau. Das Projekt befindet sich in der Planungsphase und soll zur Energiewende beitragen.",
            "source": "Lindenberg_Planungsbericht",
            "category": "general"
        },
        {
            "content": "Das Projekt hat Umweltauswirkungen, die durch Schutzmaßnahmen minimiert werden. Es gibt Kompromisse zwischen lokalen Auswirkungen und globalen Klimazielen.",
            "source": "Lindenberg_Planungsbericht", 
            "category": "environment"
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
    """Builds a prompt string"""
    prompt = []
    for name, contents in kwargs.items():
        if contents:
            prompt.append(f"<{name}>\n{contents}\n</{name}>")
    return "\n".join(prompt)

def build_question_prompt(question):
    """Create prompt with context"""
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
    """Get response from Apertus via PublicAI"""
    try:
        response = requests.post(
            "https://api.publicai.co/v1/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {st.secrets['APERTUS_KEY']}",
                "User-Agent": "WindparkChatbot/1.0"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.1,
                "top_p": 0.9
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                words = content.split()
                for i in range(0, len(words), 3):
                    yield " ".join(words[i:i+3]) + " "
                    time.sleep(0.1)
            else:
                yield "Keine Antwort vom Apertus-Modell erhalten."
        else:
            yield f"API Fehler: {response.status_code} - {response.text}"
            
    except Exception as e:
        yield f"Fehler bei der Antwortgenerierung: {str(e)}"

@st.dialog("Rechtlicher Hinweis")
def show_disclaimer_dialog():
    st.caption("""
            Dieser AI-Chatbot verwendet das Schweizer Apertus-Sprachmodell und stellt Informationen 
            über das Windpark Lindenberg Projekt bereit. Die Antworten basieren auf dem offiziellen 
            Planungsbericht. Es wird keine Rechtsberatung gegeben.
        """)

# UI
st.markdown("# 🌱 Windpark Lindenberg Assistant (Apertus)")
st.info("🇨🇭 Powered by Swiss Apertus AI Model")

title_row = st.container(horizontal=True, vertical_alignment="bottom")

with title_row:
    st.title("Windpark Lindenberg Assistant", anchor=False, width="stretch")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_just_asked_initial_question = (
    "initial_question" in st.session_state and st.session_state.initial_question
)

user_just_clicked_suggestion = (
    "selected_suggestion" in st.session_state and st.session_state.selected_suggestion
)

user_first_interaction = user_just_asked_initial_question or user_just_clicked_suggestion
has_message_history = "messages" in st.session_state and len(st.session_state.messages) > 0

if not user_first_interaction and not has_message_history:
    with st.container():
        st.chat_input("Stellen Sie eine Frage...", key="initial_question")
        selected_suggestion = st.pills(
            label="Beispiele",
            label_visibility="collapsed",
            options=SUGGESTIONS.keys(),
            key="selected_suggestion",
        )

    st.button(
        "Rechtlicher Hinweis",
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

        with st.spinner("Apertus denkt nach..."):
            full_prompt = build_question_prompt(user_message)

        with st.container():
            response = st.write_stream(get_response(full_prompt))
            st.session_state.messages.append({"role": "user", "content": user_message})
            st.session_state.messages.append({"role": "assistant", "content": response})
