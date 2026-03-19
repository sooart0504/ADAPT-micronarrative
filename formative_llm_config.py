import tomllib


class LLMConfig:

    def __init__(self, filename):

        with open(filename, "rb") as f:
            config = tomllib.load(f)

        # ── Consent ───────────────────────────────────────────────────────────
        self.intro_and_consent = config["consent"]["intro_and_consent"].strip()

        # ── Collection persona + shared settings ──────────────────────────────
        self.persona       = config["collection"]["persona"].strip()
        self.language_type = config["collection"]["language_type"].strip()
        self.topic_restriction = config["collection"]["topic_restriction"].strip()

        # ── Intro message (shown before Topic 1) ──────────────────────────────
        self.intro = config["collection"]["intro"].strip()

        # ── Per-topic questions + transitions ─────────────────────────────────
        self.topic1_questions   = config["collection"]["topic1"]["questions"]
        self.topic1_transition  = config["collection"]["topic1"]["transition"].strip()

        self.topic2_questions   = config["collection"]["topic2"]["questions"]
        self.topic2_transition  = config["collection"]["topic2"]["transition"].strip()

        self.topic3_questions   = config["collection"]["topic3"]["questions"]
        self.topic3_transition  = config["collection"]["topic3"]["transition"].strip()

        # ── Prompt templates (one per topic) ──────────────────────────────────
        self.topic1_prompt_template = self._build_question_prompt(
            config["collection"]["topic1"]["questions"], is_first_topic=True)

        self.topic2_prompt_template = self._build_question_prompt(
            config["collection"]["topic2"]["questions"], is_first_topic=False)

        self.topic3_prompt_template = self._build_question_prompt(
            config["collection"]["topic3"]["questions"], is_first_topic=False)

        # ── Extraction templates (one per topic) ──────────────────────────────
        self.topic1_extraction_template = self._build_extraction_prompt(
            config["summaries"]["topic1_questions"])
        self.topic1_keys = list(config["summaries"]["topic1_questions"].keys())

        self.topic2_extraction_template = self._build_extraction_prompt(
            config["summaries"]["topic2_questions"])
        self.topic2_keys = list(config["summaries"]["topic2_questions"].keys())

        self.topic3_extraction_template = self._build_extraction_prompt(
            config["summaries"]["topic3_questions"])
        self.topic3_keys = list(config["summaries"]["topic3_questions"].keys())

        # ── Personas (for Topic 1 story selection) ────────────────────────────
        self.personas = [p.strip() for p in config["summaries"]["personas"].values()]
        self.persona_names = list(config["summaries"]["personas"].keys())

        # ── One-shot example ──────────────────────────────────────────────────
        self.one_shot = self._build_one_shot(config["example"])

        # ── Story generation templates ────────────────────────────────────────
        self.story_prompt_template = self._build_story_prompt(
            config["summaries"]["topic1_questions"])

        self.final_story_prompt_template = self._build_final_story_prompt()

        # ── Adaptation (revision loop) ────────────────────────────────────────
        self.adaptation_prompt_template = self._build_adaptation_prompt()

        # ── Anchoring prompts ─────────────────────────────────────────────────
        self.anchoring_prompts = config["anchoring"]["prompts"]

        # ── Outros ────────────────────────────────────────────────────────────
        self.questions_outro = "Great, I think I've got everything I need — let me put your story together!"


    # ── Prompt builders ───────────────────────────────────────────────────────

    def _build_question_prompt(self, questions, is_first_topic=False):
        """Builds the conversation prompt for a given topic's question list."""

        prompt = f"{self.persona}\n\n"
        prompt += "Your goal is to gather thoughtful, heartfelt answers to the following questions:\n\n"

        for i, q in enumerate(questions):
            prompt += f"{i+1}. {q}\n"

        prompt += f"\nAsk each question one at a time. {self.language_type} "
        prompt += "Ensure you get at least a meaningful answer to each question before moving on. "
        prompt += "Never answer for the participant. "
        prompt += f"If you are unsure what they meant, gently ask again. {self.topic_restriction}"

        n = len(questions)
        if n == 1:
            prompt += "\n\nOnce you have collected an answer to the question"
        else:
            prompt += f"\n\nOnce you have collected answers to all {n} questions"

        prompt += ', stop the conversation and write a single word "FINISHED".\n\nCurrent conversation:\n{history}\nHuman: {input}\nAI:'

        return prompt


    def _build_extraction_prompt(self, questions_dict):
        """Builds the extraction prompt for a given topic's summary questions."""

        keys = list(questions_dict.keys())
        keys_string = f"`{keys[0]}`"
        for key in keys[1:-1]:
            keys_string += f", `{key}`"
        if len(keys) > 1:
            keys_string += f", and `{keys[-1]}`"

        prompt = (
            "You are an expert extraction algorithm. "
            "Only extract relevant information from the Human answers in the text. "
            "Use only the words and phrases that the text contains. "
            "If you do not know the value of an attribute asked to extract, return null for the attribute's value.\n\n"
            f"You will output a JSON with {keys_string} keys.\n\n"
            f"These correspond to the following question{'s' if len(keys) > 1 else ''}:\n"
        )

        for i, (key, question) in enumerate(questions_dict.items()):
            prompt += f"{i+1}: {question}\n"

        prompt += "\nMessage to date: {conversation_history}\n\n"
        prompt += "Remember, only extract text that is in the messages above and do not change it."

        return prompt


    def _build_one_shot(self, example):
        """Builds the one-shot example string."""
        one_shot = f"Example:\n{example['conversation']}"
        one_shot += f"\nThe story based on these responses: \"{example['scenario'].strip()}\""
        return one_shot


    def _build_story_prompt(self, questions_dict):
        """Builds the story generation prompt template used for all topics."""

        prompt = "{persona}\n\n"
        prompt += "{one_shot}\n\n"
        prompt += "Your task:\nCreate a micro-narrative story based on the following answers:\n\n"

        for key, question in questions_dict.items():
            prompt += f"Question: {question}\n"
            prompt += f"Answer: {{{key}}}\n"

        prompt += "\n{end_prompt}\n\nYour output should be a JSON file with a single entry called 'output_scenario'."

        return prompt


    def _build_final_story_prompt(self):
        """Builds the prompt for combining all 3 topic stories into one final narrative."""

        prompt = (
            "{persona}\n\n"
            "Your task is to weave together three separate micro-narrative stories about the same person "
            "into one single, coherent, and emotionally resonant story. "
            "The three stories are:\n\n"
            "Story 1 — Their ideal future self:\n{story1}\n\n"
            "Story 2 — An ideal day in their future life:\n{story2}\n\n"
            "Story 3 — The future self they fear becoming:\n{story3}\n\n"
            "Write a final combined story in first person that:\n"
            "- Begins with the person's values and who they want to become\n"
            "- Brings to life what an ideal day feels like for them\n"
            "- Acknowledges the version of themselves they are working hard not to become\n"
            "- Ends on a forward-looking, emotionally grounded note\n\n"
            "Keep the language consistent with the persona above. "
            "Your output should be a JSON file with a single entry called 'output_scenario'."
        )

        return prompt


    def _build_adaptation_prompt(self):
        """Builds the prompt for the story revision/adaptation loop."""

        prompt = (
            "You are a helpful assistant supporting someone in refining their personal story. "
            "The original story:\n\n"
            "Story: {scenario}.\n\n"
            "Their current request is: {input}.\n\n"
            "Suggest an alternative version of the story. "
            "Keep the language and content as similar as possible while fulfilling their request. "
            "Return your answer as a JSON file with a single entry called 'new_scenario'."
        )

        return prompt
