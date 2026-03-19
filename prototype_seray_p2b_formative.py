from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.json import SimpleJsonOutputParser
from langsmith import Client
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from functools import partial

import os
import sys
import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime
import streamlit.components.v1 as components

from formative_llm_config import LLMConfig

import streamlit as st


# ── Environment setup ─────────────────────────────────────────────────────────
os.environ["OPENAI_API_KEY"]        = st.secrets['OPENAI_API_KEY']
os.environ["LANGCHAIN_API_KEY"]     = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"]     = st.secrets['LANGCHAIN_PROJECT']
os.environ["LANGCHAIN_TRACING_V2"]  = 'true'

# ── Config ────────────────────────────────────────────────────────────────────
input_args  = sys.argv[1:]
config_file = input_args[0] if len(input_args) else st.secrets.get("CONFIG_FILE", "config_formative_demo.toml")
print(f"Configuring app using {config_file}...\n")

llm_prompts = LLMConfig(config_file)

# Read participant ID from URL (?pid=...)
participant_id = st.query_params.get("pid", "unknown")

DEBUG = False

smith_client = Client()

st.set_page_config(page_title="Study bot", page_icon="📖")


# ── Session state initialisation ──────────────────────────────────────────────
if 'run_id'          not in st.session_state: st.session_state['run_id']          = None
if 'created_time'    not in st.session_state: st.session_state['created_time']    = datetime.now()
if 'agentState'      not in st.session_state: st.session_state['agentState']      = "start"
if 'consent'         not in st.session_state: st.session_state['consent']         = False
if 'exp_data'        not in st.session_state: st.session_state['exp_data']        = True
if 'current_topic'   not in st.session_state: st.session_state['current_topic']   = 1
if 'locked_persona'  not in st.session_state: st.session_state['locked_persona']  = None
if 'locked_persona_name' not in st.session_state: st.session_state['locked_persona_name'] = None
if 'story_t1'        not in st.session_state: st.session_state['story_t1']        = None
if 'story_t2'        not in st.session_state: st.session_state['story_t2']        = None
if 'story_t3'        not in st.session_state: st.session_state['story_t3']        = None
if 'story_final'     not in st.session_state: st.session_state['story_final']     = None
if 'revision_history' not in st.session_state: st.session_state['revision_history'] = []
if 'answers_t1'      not in st.session_state: st.session_state['answers_t1']      = {}
if 'answers_t2'      not in st.session_state: st.session_state['answers_t2']      = {}
if 'answers_t3'      not in st.session_state: st.session_state['answers_t3']      = {}
if 'anchoring_responses' not in st.session_state: st.session_state['anchoring_responses'] = {}
if 'llm_model'       not in st.session_state: st.session_state['llm_model']       = "gpt-4o"
if 'revision_count'  not in st.session_state: st.session_state['revision_count']  = 0


# ── Memory (separate per topic) ───────────────────────────────────────────────
msgs_t1 = StreamlitChatMessageHistory(key="msgs_topic1")
msgs_t2 = StreamlitChatMessageHistory(key="msgs_topic2")
msgs_t3 = StreamlitChatMessageHistory(key="msgs_topic3")

memory_t1 = ConversationBufferMemory(memory_key="history", chat_memory=msgs_t1)
memory_t2 = ConversationBufferMemory(memory_key="history", chat_memory=msgs_t2)
memory_t3 = ConversationBufferMemory(memory_key="history", chat_memory=msgs_t3)


# ── Google Sheets save function ───────────────────────────────────────────────
def save_to_sheet():
    creds = Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    client  = gspread.authorize(creds)
    sheet   = client.open_by_key("YOUR_SHEET_ID_HERE").sheet1

    # Raw Q&A responses per topic
    t1 = st.session_state.get('answers_t1', {})
    t2 = st.session_state.get('answers_t2', {})
    t3 = st.session_state.get('answers_t3', {})

    # Revision history
    revisions = st.session_state.get('revision_history', [])
    rev1 = str(revisions[0]) if len(revisions) > 0 else ""
    rev2 = str(revisions[1]) if len(revisions) > 1 else ""

    # Anchoring responses
    anchors = st.session_state.get('anchoring_responses', {})
    anchor_values = [str(anchors.get(i, "")) for i in range(len(llm_prompts.anchoring_prompts))]

    row = [
        participant_id,
        str(st.session_state.get('created_time', "")),
        str(datetime.now()),
        str(st.session_state.get('locked_persona_name', "")),

        # Topic 1 raw answers
        *[str(t1.get(k, "")) for k in llm_prompts.topic1_keys],
        # Topic 2 raw answers
        *[str(t2.get(k, "")) for k in llm_prompts.topic2_keys],
        # Topic 3 raw answers
        *[str(t3.get(k, "")) for k in llm_prompts.topic3_keys],

        # Stories
        str(st.session_state.get('story_t1', "")),
        str(st.session_state.get('story_t2', "")),
        str(st.session_state.get('story_t3', "")),
        str(st.session_state.get('story_final', "")),

        # Revisions
        rev1,
        rev2,

        # Anchoring prompts
        *anchor_values,

        # Full chat logs
        str(msgs_t1.messages),
        str(msgs_t2.messages),
        str(msgs_t3.messages),
    ]

    sheet.append_row(row)


# ── Story extraction helper ───────────────────────────────────────────────────
def extractAnswers(msgs, keys, extraction_template):
    """Extracts structured answers from a conversation into a dict."""
    extraction_llm = ChatOpenAI(
        temperature=0.1,
        model=st.session_state.llm_model,
        openai_api_key=st.secrets.openai_api_key
    )
    template   = PromptTemplate(input_variables=["conversation_history"], template=extraction_template)
    json_parser = SimpleJsonOutputParser()
    chain      = template | extraction_llm | json_parser
    return chain.invoke({"conversation_history": msgs})


# ── Topic conversation runner ─────────────────────────────────────────────────
def runTopicConversation(topic_num, msgs, memory, prompt_template, entry_container):
    """Runs the question-asking conversation for a given topic.
    Returns True when the topic is FINISHED, False if still ongoing."""

    # Show intro message on first run of this topic
    if len(msgs.messages) == 0:
        if topic_num == 1:
            msgs.add_ai_message(llm_prompts.intro)
        else:
            transition = getattr(llm_prompts, f"topic{topic_num}_transition")
            msgs.add_ai_message(transition)

    # Show last message
    last_msgs = msgs.messages[-1:] if len(msgs.messages) >= 1 else msgs.messages
    for msg in last_msgs:
        if msg.type == "ai":
            with entry_container:
                st.chat_message("ai").write(msg.content)

    # Handle user input
    if prompt:
        chat = ChatOpenAI(
            temperature=0.3,
            model=st.session_state.llm_model,
            openai_api_key=st.secrets.openai_api_key
        )
        prompt_obj  = PromptTemplate(input_variables=["history", "input"], template=prompt_template)
        conversation = ConversationChain(prompt=prompt_obj, llm=chat, verbose=True, memory=memory)

        with entry_container:
            st.chat_message("human").write(prompt)
            response = conversation.invoke(input=prompt)

            if "FINISHED" in response['response']:
                st.divider()
                st.chat_message("ai").write(llm_prompts.questions_outro)
                return True
            else:
                st.chat_message("ai").write(response["response"])

    return False


# ── Topic 1 story generation + persona selection ──────────────────────────────
@traceable
def generateTopic1Stories():
    """Generates 3 stories for Topic 1 and lets the user pick one to lock the persona."""

    st.session_state['answers_t1'] = extractAnswers(
        msgs_t1.messages,
        llm_prompts.topic1_keys,
        llm_prompts.topic1_extraction_template
    )

    answer_set = st.session_state['answers_t1']
    summary    = {k: answer_set.get(k, "") for k in llm_prompts.topic1_keys}

    chat       = ChatOpenAI(temperature=0.7, model=st.session_state.llm_model, openai_api_key=st.secrets.openai_api_key)
    prompt_obj = PromptTemplate.from_template(llm_prompts.topic1_story_prompt_template)
    json_parser = SimpleJsonOutputParser()
    chain      = prompt_obj | chat | json_parser

    progress   = st.progress(0, "Generating your stories...")

    stories = []
    for i, persona in enumerate(llm_prompts.personas):
        story = chain.invoke({
            "persona": persona,
            "one_shot": llm_prompts.one_shot,
            "end_prompt": "Create a micro-narrative story that feels personal and emotionally resonant."
        } | summary)
        stories.append(story.get('output_scenario', ""))
        progress.progress(int((i+1) / 3 * 90), "Generating your stories...")

    progress.progress(100, "Done!")

    st.session_state['topic1_stories'] = stories
    st.session_state['agentState']     = "pick_persona"
    st.button("See my stories!", key="seeStoriesBtn")


def showPersonaPicker():
    """Shows the 3 Topic 1 stories and lets the user pick one."""

    stories = st.session_state.get('topic1_stories', [])

    st.chat_message("ai").write("Please have a look at the stories below and pick the one that best represents what you had in mind.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Story 1")
        st.write(stories[0] if len(stories) > 0 else "")
        if st.button("Pick Story 1", key="pick1"):
            lockPersona(0, stories[0])

    with col2:
        st.subheader("Story 2")
        st.write(stories[1] if len(stories) > 1 else "")
        if st.button("Pick Story 2", key="pick2"):
            lockPersona(1, stories[1])

    with col3:
        st.subheader("Story 3")
        st.write(stories[2] if len(stories) > 2 else "")
        if st.button("Pick Story 3", key="pick3"):
            lockPersona(2, stories[2])


def lockPersona(index, story):
    """Locks the chosen persona and saves Topic 1 story."""
    st.session_state['locked_persona']      = llm_prompts.personas[index]
    st.session_state['locked_persona_name'] = llm_prompts.persona_names[index]
    st.session_state['story_t1']            = story
    st.session_state['agentState']          = "topic2"
    st.session_state['current_topic']       = 2


# ── Single-topic story generator ──────────────────────────────────────────────
@traceable
def generateTopicStory(topic_num, msgs, extraction_template, keys, story_key):
    """Generates a single story for Topics 2 or 3 using the locked persona."""

    answers = extractAnswers(msgs.messages, keys, extraction_template)
    st.session_state[f'answers_t{topic_num}'] = answers
    summary = {k: answers.get(k, "") for k in keys}

    chat       = ChatOpenAI(temperature=0.7, model=st.session_state.llm_model, openai_api_key=st.secrets.openai_api_key)

    # Rebuild story prompt for this topic's questions
    topic_questions = getattr(llm_prompts, f"topic{topic_num}_questions")
    topic_summaries = {}
    for i, q in enumerate(topic_questions):
        key = keys[i] if i < len(keys) else f"q{i+1}"
        topic_summaries[key] = summary.get(key, "")

    story_template = getattr(llm_prompts, f"topic{topic_num}_story_prompt_template")
    prompt_obj     = PromptTemplate.from_template(story_template)
    json_parser    = SimpleJsonOutputParser()
    chain          = prompt_obj | chat | json_parser

    with st.spinner("Creating your story..."):
        result = chain.invoke({
            "persona":    st.session_state['locked_persona'],
            "one_shot":   llm_prompts.one_shot,
            "end_prompt": "Create a micro-narrative story that feels personal and emotionally resonant."
        } | summary)

    story = result.get('output_scenario', "")
    st.session_state[story_key] = story
    return story


# ── Final combined story ───────────────────────────────────────────────────────
@traceable
def generateFinalStory():
    """Combines all 3 topic stories into one final narrative."""

    chat        = ChatOpenAI(temperature=0.7, model=st.session_state.llm_model, openai_api_key=st.secrets.openai_api_key)
    prompt_obj  = PromptTemplate.from_template(llm_prompts.final_story_prompt_template)
    json_parser = SimpleJsonOutputParser()
    chain       = prompt_obj | chat | json_parser

    with st.spinner("Weaving your stories together..."):
        result = chain.invoke({
            "persona": st.session_state['locked_persona'],
            "story1":  st.session_state.get('story_t1', ""),
            "story2":  st.session_state.get('story_t2', ""),
            "story3":  st.session_state.get('story_t3', ""),
        })

    story = result.get('output_scenario', "")
    st.session_state['story_final']  = story
    st.session_state['agentState']   = "rate_final"
    st.rerun()


# ── Final story rating + revision ─────────────────────────────────────────────
def rateFinalStory():
    """Shows the final story and handles rating + revision (max 2 rounds)."""

    st.chat_message("ai").write("Here is your complete story, woven together from everything you shared:")
    st.markdown(f"> {st.session_state['story_final']}")
    st.divider()

    st.chat_message("ai").write("How well does this story capture what you had in mind?")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Needs some edits", key="rate_edits"):
            st.session_state['agentState'] = "revise_final"
            st.rerun()
    with col2:
        if st.button("Pretty good but I'd like to tweak it", key="rate_tweak"):
            st.session_state['agentState'] = "revise_final"
            st.rerun()
    with col3:
        if st.button("Good as is!", key="rate_good"):
            st.session_state['agentState'] = "anchoring"
            st.rerun()


def reviseFinalStory():
    """Handles the revision loop — max 2 rounds. User can type edits or edit freely."""

    revision_count = st.session_state.get('revision_count', 0)

    st.chat_message("ai").write("Here is your current story:")
    st.markdown(f"> {st.session_state['story_final']}")
    st.divider()

    if revision_count >= 2:
        st.chat_message("ai").write("You've reached the maximum number of revisions. Here is your final story!")
        st.session_state['agentState'] = "anchoring"
        st.button("Continue", key="continueAfterRevisions", on_click=lambda: st.session_state.update({'agentState': 'anchoring'}))
        return

    st.chat_message("ai").write("What would you like to change? You can type your request below, or edit the story directly.")

    # Option 1: Type a request for the chatbot to revise
    user_request = st.text_input("Type your edit request here (e.g. 'Make it more hopeful')", key=f"revision_input_{revision_count}")

    if st.button("Apply my request", key=f"apply_revision_{revision_count}") and user_request:
        chat        = ChatOpenAI(temperature=0.5, model=st.session_state.llm_model, openai_api_key=st.secrets.openai_api_key)
        prompt_obj  = PromptTemplate(input_variables=["input", "scenario"], template=llm_prompts.adaptation_prompt_template)
        json_parser = SimpleJsonOutputParser()
        chain       = prompt_obj | chat | json_parser

        with st.spinner("Revising your story..."):
            result = chain.invoke({
                'scenario': st.session_state['story_final'],
                'input':    user_request
            })

        new_story = result.get('new_scenario', st.session_state['story_final'])
        st.session_state['revision_history'].append({
            'round':    revision_count + 1,
            'request':  user_request,
            'new_story': new_story
        })
        st.session_state['story_final']    = new_story
        st.session_state['revision_count'] += 1
        st.rerun()

    st.divider()

    # Option 2: Edit the story text directly
    st.write("Or edit the story directly:")
    edited = st.text_area("Edit the story yourself", value=st.session_state['story_final'], key=f"direct_edit_{revision_count}", height=200)

    if st.button("Save my edits", key=f"save_direct_{revision_count}"):
        st.session_state['revision_history'].append({
            'round':    revision_count + 1,
            'request':  "Direct edit by participant",
            'new_story': edited
        })
        st.session_state['story_final']    = edited
        st.session_state['revision_count'] += 1
        st.session_state['agentState']     = "anchoring"
        st.rerun()


# ── Anchoring prompts ─────────────────────────────────────────────────────────
def showAnchoringPrompts():
    """Shows the final story and the sentence-completion anchoring form."""

    st.chat_message("ai").write("Here is your final story:")
    st.markdown(f"> {st.session_state['story_final']}")
    st.divider()

    st.chat_message("ai").write(
        "Finally, take a moment to complete these sentences. "
        "There are no right or wrong answers — just write what feels true for you."
    )

    responses = {}
    for i, prompt_text in enumerate(llm_prompts.anchoring_prompts):
        response = st.text_input(
            label=f"{prompt_text}...",
            key=f"anchor_{i}",
            placeholder="Type your answer here"
        )
        responses[i] = response

    if st.button("Submit and finish", key="submitAnchoring"):
        # Check all prompts have at least something
        if all(responses[i].strip() for i in range(len(llm_prompts.anchoring_prompts))):
            st.session_state['anchoring_responses'] = responses
            st.session_state['agentState'] = "complete"
            st.rerun()
        else:
            st.warning("Please complete all sentences before submitting.")


# ── Completion ────────────────────────────────────────────────────────────────
def completeSession():
    """Saves everything to Google Sheets and closes the tab."""

    with st.spinner("Saving your responses..."):
        save_to_sheet()

    st.success("Thank you! Your responses have been saved.")
    st.markdown("You've completed the activity. You can now close this tab and return to the survey.")
    components.html(
        '<script>setTimeout(function(){ window.close(); }, 3000);</script>'
    )


# ── Main state agent ──────────────────────────────────────────────────────────
def stateAgent():
    """Controls the full flow of the app based on agentState."""

    state = st.session_state['agentState']

    # ── Topic 1 ──
    if state == "start":
        finished = runTopicConversation(
            1, msgs_t1, memory_t1,
            llm_prompts.topic1_prompt_template,
            entry_messages
        )
        if finished:
            st.session_state['agentState'] = "gen_t1_stories"
            generateTopic1Stories()

    elif state == "gen_t1_stories":
        generateTopic1Stories()

    elif state == "pick_persona":
        showPersonaPicker()

    # ── Topic 2 ──
    elif state == "topic2":
        finished = runTopicConversation(
            2, msgs_t2, memory_t2,
            llm_prompts.topic2_prompt_template,
            entry_messages
        )
        if finished:
            story = generateTopicStory(
                2, msgs_t2,
                llm_prompts.topic2_extraction_template,
                llm_prompts.topic2_keys,
                'story_t2'
            )
            st.divider()
            st.chat_message("ai").write("Here is your story for this part:")
            st.markdown(f"> {story}")
            st.session_state['agentState'] = "topic3"
            st.button("Continue to next part", key="toTopic3")

    # ── Topic 3 ──
    elif state == "topic3":
        finished = runTopicConversation(
            3, msgs_t3, memory_t3,
            llm_prompts.topic3_prompt_template,
            entry_messages
        )
        if finished:
            story = generateTopicStory(
                3, msgs_t3,
                llm_prompts.topic3_extraction_template,
                llm_prompts.topic3_keys,
                'story_t3'
            )
            st.divider()
            st.chat_message("ai").write("Here is your story for this part:")
            st.markdown(f"> {story}")
            st.session_state['agentState'] = "gen_final"
            st.button("See my complete story", key="toFinal")

    # ── Final story ──
    elif state == "gen_final":
        generateFinalStory()

    elif state == "rate_final":
        rateFinalStory()

    elif state == "revise_final":
        reviseFinalStory()

    # ── Anchoring ──
    elif state == "anchoring":
        showAnchoringPrompts()

    # ── Complete ──
    elif state == "complete":
        completeSession()


# ── Consent gate ──────────────────────────────────────────────────────────────
def markConsent():
    st.session_state['consent'] = True


st.markdown("""
    <style>
    [data-testid="stToolbarActions"] {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


if st.session_state['consent']:
    entry_messages = st.expander("Our conversation", expanded=st.session_state['exp_data'])
    prompt = st.chat_input()

    if "openai_api_key" in st.secrets:
        openai_api_key = st.secrets.openai_api_key
    else:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Enter an OpenAI API Key to continue")
        st.stop()

    stateAgent()

else:
    consent_message = st.container()
    with consent_message:
        st.markdown(llm_prompts.intro_and_consent)
        st.button("I accept", key="consent_button", on_click=markConsent)
