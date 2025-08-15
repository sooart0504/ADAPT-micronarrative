

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langsmith import Client
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from streamlit_feedback import streamlit_feedback

from functools import partial

import os
import sys

from llm_config import LLMConfig

import streamlit as st


# Using streamlit secrets to set environment variables for langsmith/chain
os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']
os.environ["LANGCHAIN_API_KEY"] = st.secrets['LANGCHAIN_API_KEY']
os.environ["LANGCHAIN_PROJECT"] = st.secrets['LANGCHAIN_PROJECT']
os.environ["LANGCHAIN_TRACING_V2"] = 'true'


# Parse input args, checking for config file
input_args = sys.argv[1:]
if len(input_args):
    config_file = input_args[0]
else:
    config_file = st.secrets.get("CONFIG_FILE", "example_config.toml")
print(f"Configuring app using {config_file}...\n")

# Create prompts based on configuration file
llm_prompts = LLMConfig(config_file)

## simple switch previously used to help debug 
DEBUG = False

# Langsmith set-up 
smith_client = Client()


st.set_page_config(page_title="Study bot", page_icon="📖")
# st.title("📖 Study bot")


## initialising key variables in st.sessionstate if first run
if 'run_id' not in st.session_state: 
    st.session_state['run_id'] = None

if 'agentState' not in st.session_state: 
    st.session_state['agentState'] = "start"
if 'consent' not in st.session_state: 
    st.session_state['consent'] = False
if 'exp_data' not in st.session_state: 
    st.session_state['exp_data'] = True

## set the model to use in case this is the first run 
if 'llm_model' not in st.session_state:
    # st.session_state.llm_model = "gpt-3.5-turbo-1106"
    st.session_state.llm_model = "gpt-4o"

# Set up memory for the lanchchain conversation bot
msgs = StreamlitChatMessageHistory(key="langchain_messages")
memory = ConversationBufferMemory(memory_key="history", chat_memory=msgs)



# selections = st.sidebar


# with selections:
#     st.markdown("## LLM model selection")
#     st.markdown(":blue[Different models have widely differing costs.   \n \n  It seems that running this whole flow with chatGPT 4 costs about $0.1 per full flow as there are multiple processing steps 👻; while the 3.5-turbo is about 100x cheaper 🤑 and gpt-4o is about 6x cheaper than gpt4.]")
#     st.markdown('**Our prompts are currently set up for gpt-4o so you might want to run your first trial with that** ... however, multiple runs might be good to with some of the cheaper models.')
    


#     st.session_state.llm_model = st.selectbox(
#         "Which LLM would you like to try?",
#         [ 
#             'gpt-4o', 
#             'gpt-4',
#             'gpt-3.5-turbo-1106'
#             ],
#         key = 'llm_choice',
#     )

#     st.write("**Current llm-model selection:**  \n " + st.session_state.llm_model)


## ensure we are using a better prompt for 4o 
if st.session_state['llm_model'] == "gpt-4o":
    prompt_datacollection = llm_prompts.questions_prompt_template



def getData (testing = False ): 
    """Collects answers to main questions from the user. 
    
    The conversation flow is stored in the msgs variable (which acts as the persistent langchain-streamlit memory for the bot). The prompt for LLM must be set up to return "FINISHED" when all data is collected. 
    
    Parameters: 
    testing: bool variable that will insert a dummy conversation instead of engaging with the user

    Returns: 
    Nothing returned as all data is stored in msgs. 
    """

    ## if this is the first run, set up the intro 
    if len(msgs.messages) == 0:
        msgs.add_ai_message(llm_prompts.questions_intro)


   # as Streamlit refreshes page after each input, we have to refresh all messages. 
   # in our case, we are just interested in showing the last AI-Human turn of the conversation for simplicity

    if len(msgs.messages) >= 2:
        last_two_messages = msgs.messages[-1:]
    else:
        last_two_messages = msgs.messages

    for msg in last_two_messages:
        if msg.type == "ai":
            with entry_messages:
                st.chat_message(msg.type).write(msg.content)


    # If user inputs a new answer to the chatbot, generate a new response and add into msgs
    if prompt:
        # Note: new messages are saved to history automatically by Langchain during run 
        with entry_messages:
            # show that the message was accepted 
            st.chat_message("human").write(prompt)
            
            # generate the reply using langchain 
            response = conversation.invoke(input = prompt)
            
            # the prompt must be set up to return "FINISHED" once all questions have been answered
            # If finished, move the flow to summarisation, otherwise continue.
            if "FINISHED" in response['response']:
                st.divider()
                st.chat_message("ai").write(llm_prompts.questions_outro)

                # call the summarisation  agent
                st.session_state.agentState = "summarise"
                summariseData(testing)
            else:
                st.chat_message("ai").write(response["response"])

 
        
        #st.text(st.write(response))


def extractChoices(msgs, testing ):
    """Uses bespoke LLM prompt to extract answers to given questions from a conversation history into a JSON object. 

    Arguments: 
    msgs (str): conversations history to extract from -- this can be streamlit memory, or a dummy variable during testing
    testing (bool): bool variable that will insert a dummy conversation instead of engaging with the user

    """

    ## set up our extraction LLM -- low temperature for repeatable results
    extraction_llm = ChatOpenAI(temperature=0.1, model=st.session_state.llm_model, openai_api_key=openai_api_key)

    ## taking the prompt from lc_prompts.py file
    extraction_template = PromptTemplate(input_variables=["conversation_history"], template = llm_prompts.extraction_prompt_template)

    ## set up the rest of the chain including the json parser we will need. 
    json_parser = SimpleJsonOutputParser()
    extractionChain = extraction_template | extraction_llm | json_parser

    
    # allow for testing the flow with pre-generated messages -- see testing_prompts.py
    if testing:
        extractedChoices = extractionChain.invoke({"conversation_history" : llm_prompts.example_messages})
    else: 
        extractedChoices = extractionChain.invoke({"conversation_history" : msgs})
    

    return(extractedChoices)


def collectFeedback(answer, column_id,  scenario):
    """ Submits user's feedback on specific scenario to langsmith; called as on_submit function for the respective streamlit feedback object. 
    
    The payload combines the text of the scenario, user output, and answers. This function is intended to be called as 'on_submit' for the streamlit_feedback component.  

    Parameters: 
    answer (dict): Returned by streamlit_feedback function, contains "the user response, with the feedback_type, score and text fields" 
    column_id (str): marking which column this belong too 
    scenario (str): the scenario that users submitted feedback on

    """

    st.session_state.temp_debug = "called collectFeedback"
    
    # allows us to pick between thumbs / faces, based on the streamlit_feedback response
    score_mappings = {
        "thumbs": {"👍": 1, "👎": 0},
        "faces": {"😀": 1, "🙂": 0.75, "😐": 0.5, "🙁": 0.25, "😞": 0},
    }
    scores = score_mappings[answer['type']]
    
    # Get the score from the selected feedback option's score mapping
    score = scores.get(answer['score'])

    # store the Langsmith run_id so the feedback is attached to the right flow on Langchain side 
    run_id = st.session_state['run_id']

    if DEBUG: 
        st.write(run_id)
        st.write(answer)


    if score is not None:
        # Formulate feedback type string incorporating the feedback option
        # and score value
        feedback_type_str = f"{answer['type']} {score} {answer['text']} \n {scenario}"

        st.session_state.temp_debug = feedback_type_str

        ## combine all data that we want to store in Langsmith
        payload = f"{answer['score']} rating scenario: \n {scenario} \n Based on: \n {llm_prompts.one_shot}"

        # Record the feedback with the formulated feedback type string
        # and optional comment
        smith_client.create_feedback(
            run_id= run_id,
            value = payload,
            key = column_id,
            score=score,
            comment=answer['text']
        )
    else:
        st.warning("Invalid feedback score.")    



@traceable # Auto-trace this function
def summariseData(testing = False): 
    """Takes the extracted answers to questions and generates three scenarios, based on selected prompts. 

    testing (bool): will insert a dummy data instead of user-generated content if set to True

    """


    # start by setting up the langchain chain from our template (defined in lc_prompts.py)
    prompt_template = PromptTemplate.from_template(llm_prompts.main_prompt_template)

    # add a json parser to make sure the output is a json object
    json_parser = SimpleJsonOutputParser()

    # connect the prompt with the llm call, and then ensure output is json with our new parser
    chain = prompt_template | chat | json_parser

    ### call extract choices on real data / stored test data based on value of testing
    if testing: 
        answer_set = extractChoices(msgs, True)
    else:
        answer_set = extractChoices(msgs, False)
    
    ## debug shows the interrim steps of the extracted set
    if DEBUG: 
        st.divider()
        st.chat_message("ai").write("**DEBUGGING** *-- I think this is a good summary of what you told me ... check if this is correct!*")
        st.chat_message("ai").json(answer_set)

    # store the generated answers into streamlit session state
    st.session_state['answer_set'] = answer_set


    # let the user know the bot is starting to generate content 
    with entry_messages:
        if testing:
            st.markdown(":red[DEBUG active -- using testing messages]")

        st.divider()
        st.chat_message("ai").write("Seems I have everything! Let me try to summarise what you said in three scenarios. \n See you if you like any of these! ")


        ## can't be bothered to set up LLM stream here, so just showing progress bar for now  
        ## this gets manually updated after each scenario
        progress_text = 'Processing your scenarios'
        bar = st.progress(0, text = progress_text)

    # Arrange answers into dictionary
    summary_answers = {key: answer_set[key] for key in llm_prompts.summary_keys}

    # create first scenario & store into st.session state 
    st.session_state.response_1 = chain.invoke({
        "persona" : llm_prompts.personas[0],
        "one_shot": llm_prompts.one_shot,
        "end_prompt" : llm_prompts.extraction_task} | summary_answers)
    run_1 = get_current_run_tree()

    ## update progress bar
    bar.progress(33, progress_text)

    st.session_state.response_2 = chain.invoke({
        "persona" : llm_prompts.personas[1],
        "one_shot": llm_prompts.one_shot,
        "end_prompt" : llm_prompts.extraction_task} | summary_answers)
    run_2 = get_current_run_tree()

    ## update progress bar
    bar.progress(66, progress_text)

    st.session_state.response_3 = chain.invoke({
        "persona" : llm_prompts.personas[2],
        "one_shot": llm_prompts.one_shot,
        "end_prompt" : llm_prompts.extraction_task} | summary_answers)
    run_3 = get_current_run_tree()

    ## update progress bar after the last scenario
    bar.progress(99, progress_text)

    # remove the progress bar
    # bar.empty()

    if DEBUG: 
        st.session_state.run_collection = {
            "run1": run_1,
            "run2": run_2,
            "run3": run_3
        }

    ## update the correct run ID -- all three calls share the same one. 
    st.session_state.run_id = run_1.id

    ## move the flow to the next state
    st.session_state["agentState"] = "review"

    # we need the user to do an action (e.g., button click) to generate a natural streamlit refresh (so we can show scenarios on a clear page). Other options like streamlit rerun() have been marked as 'failed runs' on Langsmith which is annoying. 
    st.button("I'm ready -- show me!", key = 'progressButton')


def testing_reviewSetUp():
    """Simple function that just sets up dummy scenario data, used when testing later flows of the process. 
    """
    

    ## setting up testing code -- will likely be pulled out into a different procedure 
    text_scenarios = {
        "s1" : "So, here's the deal. I've been really trying to get my head around this coding thing, specifically in langchain. I thought I'd share my struggle online, hoping for some support or advice. But guess what? My PhD students and postdocs, the very same people I've been telling how crucial it is to learn coding, just laughed at me! Can you believe it? It made me feel super ticked off and embarrassed. I mean, who needs that kind of negativity, right? So, I did what I had to do. I let all the postdocs go, re-advertised their positions, and had a serious chat with the PhDs about how uncool their reaction was to my coding struggles.",

        "s2": "So, here's the thing. I've been trying to learn this coding thing called langchain, right? It's been a real struggle, so I decided to share my troubles online. I thought my phd students and postdocs would understand, but instead, they just laughed at me! Can you believe that? After all the times I've told them how important it is to learn how to code. It made me feel really mad and embarrassed, you know? So, I did what I had to do. I told the postdocs they were out and had to re-advertise their positions. And I had a serious talk with the phds, telling them that laughing at my coding struggles was not cool at all.",

        "s3": "So, here's the deal. I've been trying to learn this coding language called langchain, right? And it's been a real struggle. So, I decided to post about it online, hoping for some support or advice. But guess what? My PhD students and postdocs, the same people I've been telling how important it is to learn coding, just laughed at me! Can you believe it? I was so ticked off and embarrassed. I mean, who does that? So, I did what any self-respecting person would do. I fired all the postdocs and re-advertised their positions. And for the PhDs? I had a serious talk with them about how uncool their reaction was to my coding struggles."
    }

    # insert the dummy text into the right st.sessionstate locations 
    st.session_state.response_1 = {'output_scenario': text_scenarios['s1']}
    st.session_state.response_2 = {'output_scenario': text_scenarios['s2']}
    st.session_state.response_3 = {'output_scenario': text_scenarios['s3']}


def click_selection_yes(button_num, scenario):
    """ Function called on_submit when a final scenario is selected. 
    
    Saves all key information in the st.session_state.scenario_package persistent variable.
    """
    st.session_state.scenario_selection = button_num
    
    ## if we are testing, the answer_set might not have been set & needs to be added:
    if 'answer_set' not in st.session_state:
        st.session_state['answer_set'] = "Testing - no answers"

    ## save all important information in one package into st.session state

    scenario_dict = {
        'col1': st.session_state.response_1['output_scenario'],
        'col2': st.session_state.response_2['output_scenario'],
        'col3': st.session_state.response_3['output_scenario'],
        'fb1': st.session_state['col1_fb'],
        'fb2': st.session_state['col2_fb'],
        'fb3': st.session_state['col3_fb']
    }

    st.session_state.scenario_package = {
            'scenario': scenario,
            'answer set':  st.session_state['answer_set'],
            'judgment': st.session_state['scenario_decision'],
            'scenarios_all': scenario_dict,
            'chat history': msgs
    }


def click_selection_no():
    """ Function called on_submit when a user clicks on 'actually, let me try another one'. 
     
    The only purpose is to set the scenario judged flag back on 
    """
    st.session_state['scenario_judged'] = True

def sliderChange(name, *args):
    """Function called on_change for the 'Judge_scenario' slider.  
    
    It updates two variables:
    st.session_state['scenario_judged'] -- which shows that some rating was provided by the user and un-disables a button for them to accept the scenario and continue 
    st.session_state['scenario_decision'] -- which stores the current rating

    """
    st.session_state['scenario_judged'] = False
    st.session_state['scenario_decision'] = st.session_state[name]


     
def scenario_selection (popover, button_num, scenario):
    """ Helper function which sets up the text & infrastructure for each scenario popover. 

    Arguments: 
    popover: streamlit popover object that we are operating on 
    button_num (str): allows us to keep track which scenario column the popover belongs to 
    scenario (str): the text of the scenario that the button refers to  
    """
    with popover:
        
        ## if this is the first run, set up the scenario_judged flag -- this will ensure that people cannot accept a scenario without rating it first (by being passes as the argument into 'disabled' option of the c1.button). For convenience and laziness, the bool is flipped -- "True" here means that 'to be judged'; "False" is 'has been judged'. 
        if "scenario_judged" not in st.session_state:
            st.session_state['scenario_judged'] = True


        st.markdown(f"How well does the scenario {button_num} capture what you had in mind?")
        sliderOptions = ["Not really ", "Needs some edits", "Pretty good but I'd like to tweak it", "Ready as is!"]
        slider_name = f'slider_{button_num}'

        st.select_slider("Judge_scenario", label_visibility= 'hidden', key = slider_name, options = sliderOptions, on_change= sliderChange, args = (slider_name,))
        

        c1, c2 = st.columns(2)
        
        ## the accept button should be disabled if no rating has been provided yet
        c1.button("Continue with this scenario 🎉", key = f'yeskey_{button_num}', on_click = click_selection_yes, args = (button_num, scenario), disabled = st.session_state['scenario_judged'])

        ## the second one needs to be accessible all the time!  
        c2.button("actually, let me try another one 🤨", key = f'nokey_{button_num}', on_click= click_selection_no)



def reviewData(testing):
    """ Procedure that governs the scenario review and selection by the user. 

    It presents the scenarios generated in previous phases (and saved to st.session_state) and sets up the feedback / selection buttons and popovers. 
    """

    ## If we're testing this function, the previous functions have set up the three column structure yet and we don't have scenarios. 
    ## --> we will set these up now. 
    if testing:
        testing_reviewSetUp() 


    ## if this is the first time running, let's make sure that the scenario selection variable is ready. 
    if 'scenario_selection' not in st.session_state:
        st.session_state['scenario_selection'] = '0'

    ## assuming no scenario has been selected 
    if st.session_state['scenario_selection'] == '0':
        # setting up space for the scenarios 
        col1, col2, col3 = st.columns(3)
        
        ## check if we had any feedback before:
        ## set up a dictionary:
        disable = {
            'col1_fb': None,
            'col2_fb': None,
            'col3_fb': None,
        }
        ## grab any answers we already have:
        for col in ['col1_fb','col2_fb','col3_fb']:
            if col in st.session_state and st.session_state[col] is not None:
                
                if DEBUG: 
                    st.write(col)
                    st.write("Feeedback 1:", st.session_state[col]['score'])
                
                # update the corresponding entry in the disable dict
                disable[col] = st.session_state[col]['score']

        # now set up the columns with each scenario & feedback functions
        with col1: 
            st.header("Scenario 1") 
            st.write(st.session_state.response_1['output_scenario'])
            col1_fb = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                align='center',
                key="col1_fb",
                # this ensures that feedback cannot be submitted twice 
                disable_with_score = disable['col1_fb'],
                on_submit = collectFeedback,
                args = ('col1',
                        st.session_state.response_1['output_scenario']
                        )
            )

        with col2: 
            st.header("Scenario 2") 
            st.write(st.session_state.response_2['output_scenario'])
            col2_fb = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                align='center',
                key="col2_fb",
                # this ensures that feedback cannot be submitted twice 
                disable_with_score = disable['col2_fb'],            
                on_submit = collectFeedback,
                args = ('col2', 
                        st.session_state.response_2['output_scenario']
                        )
            )        
        
        with col3: 
            st.header("Scenario 3") 
            st.write(st.session_state.response_3['output_scenario'])
            col3_fb = streamlit_feedback(
                feedback_type="thumbs",
                optional_text_label="[Optional] Please provide an explanation",
                align='center',
                key="col3_fb",
                # this ensures that feedback cannot be submitted twice 
                disable_with_score = disable['col3_fb'],            
                on_submit = collectFeedback,
                args = ('col3', 
                        st.session_state.response_3['output_scenario']
                        )
            )   


        ## now we should have col1, col2, col3 with text available -- let's set up the infrastructure for selection. 
        st.divider()

        if DEBUG:
            st.write("run ID", st.session_state['run_id'])
            if 'temp_debug' not in st.session_state:
                st.write("no debug found")
            else:
                st.write("debug feedback", st.session_state.temp_debug)
        


        ## if we haven't selected scenario, let's give them a choice. 
        st.chat_message("ai").write("Please have a look at the scenarios above. Use the 👍 and 👎  to leave a rating and short comment on each of the scenarios. Then pick the one that you like the most to continue. ")
     
        b1,b2,b3 = st.columns(3)
        # set up the popover buttons 
        p1 = b1.popover('Pick scenario 1', use_container_width=True)
        p2 = b2.popover('Pick scenario 2', use_container_width=True)
        p3 = b3.popover('Pick scenario 3', use_container_width=True)

        # and now initialise them properly
        scenario_selection(p1,'1', st.session_state.response_1['output_scenario']) 
        scenario_selection(p2,'2',st.session_state.response_2['output_scenario']) 
        scenario_selection(p3,'3',st.session_state.response_3['output_scenario']) 
    
    
    ## and finally, assuming we have selected a scenario, let's move into the final state!  Note that we ensured that the screen is free for any new content now as people had to click to select a scenario -- streamlit is starting with a fresh page 
    else:
        # great, we have a scenario selected, and all the key information is now in st.session_state['scenario_package'], created in the def click_selection_yes(button_num, scenario):

        # set the flow pointer accordingly 
        st.session_state['agentState'] = 'finalise'
        # print("ended loop -- should move to finalise!")
        finaliseScenario()


def updateFinalScenario (new_scenario):
    """ Updates the final scenario when the user accepts. 
    """
    st.session_state.scenario_package['scenario'] = new_scenario
    st.session_state.scenario_package['judgment'] = "Ready as is!"

@traceable
def finaliseScenario():
    """ Procedure governs the last part of the flow, which is the scenario adaptation.
    """

    # grab a 'local' copy of the package collected in the previous flow
    package = st.session_state['scenario_package']

    # if scenario is judged as 'ready' by the user -- we're done
    if package['judgment'] == "Ready as is!":
        st.markdown(":tada: Yay! :tada:")
        st.markdown("You've now completed the interaction and hopefully found a scenario that you liked! ")
        st.markdown(f":green[{package['scenario']}]")
    
    
    # if the user still wants to continue adapting
    else:
        # set up a streamlit container for the original scenario
        original = st.container()
        
        with original:
            st.markdown(f"It seems that you selected a scenario that you liked: \n\n :green[{package['scenario']}]")
            st.markdown(f"... but that you also think it: :red[{package['judgment']}]")


        # set up a streamlit container for the new conversation & adapted scenario
        adapt_convo_container = st.container()
        
        with adapt_convo_container:
            st.chat_message("ai").write("Okay, what's missing or could change to make this better?")
        
            # once user enters something 
            if prompt:
                st.chat_message("human").write(prompt) 

                # use a new chain, drawing on the prompt_adaptation template from lc_prompts.py
                adaptation_prompt = PromptTemplate(input_variables=["input", "scenario"], template = llm_prompts.extraction_adaptation_prompt_template)
                json_parser = SimpleJsonOutputParser()

                chain = adaptation_prompt | chat | json_parser

                # set up a UX feedback in case the scenario takes longer to generate
                # note -- spinner disappears once the code inside finishes
                with st.spinner('Working on your updated scenario 🧐'):
                    new_response = chain.invoke({
                        'scenario': package['scenario'], 
                        'input': prompt
                        })
                    # st.write(new_response)

                st.markdown(f"Here is the adapted response: \n :orange[{new_response['new_scenario']}]\n\n **what do you think?**")

              
                c1, c2  = st.columns(2)

                c1.button("All good!", 
                          on_click=updateFinalScenario,
                          args=(new_response['new_scenario'],))

                # clicking the "keep adapting" button will force streamlit to refresh the page 
                # --> this loop will run again.  
                c2.button("Keep adapting")


                ## TODO -- add an opportunity for people to rewrite the scenario themselves. 
                # The implementation below wasn't very aesthetically pleasing. 

                # popover_rewrite = c3.popover("I'll rewrite it myself")
                # with popover_rewrite:
                #     txt = st.text_area("Edit the scenario yourself and press command + Enter when you're happy with it",value=new_response['new_scenario'], on_change=test_area)            


            

def stateAgent(): 
    """ Main flow function of the whole interaction -- keeps track of the system state and calls the appropriate procedure on each streamlit refresh. 
    """

    # testing will ensure using dummy data (rather than user-data collection) to simplify development / testing of later parts of the flow. 
    testing = False

    # keep track of where we are, if testing
    if testing:
        print("Running stateAgent loop -- session state: ", st.session_state['agentState'])


    # Main loop -- selecting the right 'agent' each time: 
    if st.session_state['agentState'] == 'start':
            getData(testing)
            # summariseData(testing)
            # reviewData(testing)
    elif st.session_state['agentState'] == 'summarise':
            summariseData(testing)
    elif st.session_state['agentState'] == 'review':
            reviewData(testing)
    elif st.session_state['agentState'] == 'finalise':
            finaliseScenario()



def markConsent():
    """On_submit function that marks the consent progress 
    """
    st.session_state['consent'] = True



## hide the github icon so we don't de-anonymise! 
st.markdown(
"""
    <style>
    [data-testid="stToolbarActions"] {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)
### check we have consent -- if so, run normally 
if st.session_state['consent']: 
    
    # setting up the right expanders for the start of the flow
    if st.session_state['agentState'] == 'review':
        st.session_state['exp_data'] = False

    entry_messages = st.expander("Collecting your story", expanded = st.session_state['exp_data'])

    if st.session_state['agentState'] == 'review':
        review_messages = st.expander("Review Scenarios")

    
    # create the user input object 
    prompt = st.chat_input()


    # Get an OpenAI API Key before continuing
    if "OPENAI_API_KEY" in st.secrets:
        openai_api_key = st.secrets.openai_api_key
    else:
        openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    if not openai_api_key:
        st.info("Enter an OpenAI API Key to continue")
        st.stop()


    # Set up the LangChain for data collection, passing in Message History
    chat = ChatOpenAI(temperature=0.3, model=st.session_state.llm_model, openai_api_key = openai_api_key)

    prompt_updated = PromptTemplate(input_variables=["history", "input"], template = prompt_datacollection)

    conversation = ConversationChain(
        prompt = prompt_updated,
        llm = chat,
        verbose = True,
        memory = memory
        )
    
    # start the flow agent 
    stateAgent()

# we don't have consent yet -- ask for agreement and wait 
else: 
    print("don't have consent!")
    consent_message = st.container()
    with consent_message:
        st.markdown(llm_prompts.intro_and_consent)
        st.button("I accept", key = "consent_button", on_click=markConsent)
           



