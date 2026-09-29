import os

import requests
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

Question = io.Custom("SYSTEMONE_QUESTION")
Answers = io.Custom("SYSTEMONE_ANSWERS")

MISSING = object()
CHOICE_SWITCH_OPTIONS = 10


def parse_criteria(question_type, text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if question_type == "score":
        return lines
    criteria = {}
    for line in lines:
        key, sep, description = line.partition(":")
        if not sep:
            raise ValueError(f"Criteria line must be 'key: description', got {line!r}")
        criteria[key.strip()] = description.strip()
    return criteria


class SystemOneQuestion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SystemOneQuestion",
            display_name="SystemOne Question",
            category="SystemOne",
            inputs=[
                io.String.Input("name", default="question"),
                io.Combo.Input("type", options=["choice", "score", "noul"]),
                io.String.Input("instructions", multiline=True),
                io.String.Input(
                    "criteria",
                    multiline=True,
                    tooltip="choice: one 'key: description' per line. score: one level per line, lowest first. noul: optional 'true: ...' and 'false: ...' lines.",
                ),
            ],
            outputs=[Question.Output()],
        )

    @classmethod
    def execute(cls, name, type, instructions, criteria) -> io.NodeOutput:
        question = {"type": type, "instructions": instructions}
        parsed = parse_criteria(type, criteria)
        if parsed:
            question["criteria"] = parsed
        return io.NodeOutput((name, question))


class SystemOne(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SystemOne",
            display_name="SystemOne",
            category="SystemOne",
            description="Ask a SystemOne server questions about a state. Set TYPESAFE_API_KEY to authenticate with a hosted server.",
            inputs=[
                io.String.Input("server_url", default="http://localhost:8765/v1/systemone"),
                io.String.Input("model", default="jeff-latest"),
                io.String.Input("state", multiline=True),
                io.Autogrow.Input(
                    "questions",
                    template=io.Autogrow.TemplatePrefix(Question.Input("question"), prefix="question", min=1, max=50),
                ),
            ],
            outputs=[Answers.Output()],
        )

    @classmethod
    def execute(cls, server_url, model, state, questions) -> io.NodeOutput:
        request_questions = {}
        for name, question in questions.values():
            if name in request_questions:
                raise ValueError(f"Duplicate SystemOne question name: {name!r}")
            request_questions[name] = question

        headers = {}
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        response = requests.post(
            server_url,
            json={"model": model, "state": state, "questions": request_questions},
            headers=headers,
            timeout=60,
        )
        if not response.ok:
            raise RuntimeError(f"SystemOne request failed ({response.status_code}): {response.text}")
        answers = response.json()["answers"]
        return io.NodeOutput({name: {"question": question, "answer": answers[name]} for name, question in request_questions.items()})


def answer_outputs(question, answer):
    if answer["type"] == "choice":
        keys = list(question["criteria"])
        choice = answer["choice"]
        probabilities = [answer["probabilities"].get(key, 0.0) for key in keys]
        return choice, keys.index(choice), answer["probabilities"][choice], answer["confidence"], probabilities
    if answer["type"] == "score":
        score = answer["score"]
        probabilities = [answer["probabilities"].get(str(i), 0.0) for i in range(len(question["criteria"]))]
        return str(round(score)), round(score), score, answer["confidence"], probabilities
    noul = answer["noul"]
    yes = noul >= 0.5
    return ("true" if yes else "false"), int(yes), noul, 1.0, [1.0 - noul, noul]


class SystemOneAnswer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SystemOneAnswer",
            display_name="SystemOne Answer",
            category="SystemOne",
            inputs=[
                Answers.Input("answers"),
                io.String.Input("name", default="question"),
            ],
            outputs=[
                io.String.Output(display_name="choice", tooltip="choice: option key. score: rounded level. noul: 'true' if P(yes) >= 0.5."),
                io.Int.Output(display_name="index", tooltip="choice: position of the option in criteria. score: rounded level. noul: 1 or 0."),
                io.Float.Output(display_name="value", tooltip="choice: probability of the chosen option. score: weighted score. noul: P(yes)."),
                io.Float.Output(display_name="confidence", tooltip="Answer concentration. Noul has no confidence and always outputs 1.0."),
                io.Float.Output(display_name="probabilities", tooltip="Per option or level, in criteria order. noul: [P(no), P(yes)].", is_output_list=True),
            ],
        )

    @classmethod
    def execute(cls, answers, name) -> io.NodeOutput:
        if name not in answers:
            raise ValueError(f"No SystemOne answer named {name!r}. Available: {', '.join(answers)}")
        return io.NodeOutput(*answer_outputs(answers[name]["question"], answers[name]["answer"]))


class SystemOneThresholdSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="SystemOneThresholdSwitch",
            display_name="SystemOne Threshold Switch",
            category="SystemOne",
            description="Outputs on_true when value >= threshold, otherwise on_false. Only the selected branch runs.",
            inputs=[
                io.Float.Input("value", force_input=True),
                io.Float.Input("threshold", default=0.8, min=0.0, max=10.0, step=0.01),
                io.MatchType.Input("on_true", template=template, lazy=True, optional=True),
                io.MatchType.Input("on_false", template=template, lazy=True, optional=True),
            ],
            outputs=[io.MatchType.Output(template=template, display_name="output")],
        )

    @classmethod
    def check_lazy_status(cls, value, threshold, on_true=MISSING, on_false=MISSING):
        if value >= threshold and on_true is None:
            return ["on_true"]
        if value < threshold and on_false is None:
            return ["on_false"]

    @classmethod
    def execute(cls, value, threshold, on_true=MISSING, on_false=MISSING) -> io.NodeOutput:
        selected = on_true if value >= threshold else on_false
        return io.NodeOutput(None if selected is MISSING else selected)


class SystemOneChoiceSwitch(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        template = io.MatchType.Template("switch")
        return io.Schema(
            node_id="SystemOneChoiceSwitch",
            display_name="SystemOne Choice Switch",
            category="SystemOne",
            description="Outputs option{index}. Only the selected option runs. Chain another switch for more than 10 options.",
            inputs=[io.Int.Input("index", default=0, min=0, max=CHOICE_SWITCH_OPTIONS - 1)]
            + [io.MatchType.Input(f"option{i}", template=template, lazy=True, optional=True) for i in range(CHOICE_SWITCH_OPTIONS)],
            outputs=[io.MatchType.Output(template=template, display_name="output")],
        )

    @classmethod
    def check_lazy_status(cls, index, **options):
        key = f"option{index}"
        if key in options and options[key] is None:
            return [key]

    @classmethod
    def execute(cls, index, **options) -> io.NodeOutput:
        key = f"option{index}"
        if key not in options:
            raise ValueError(f"SystemOne Choice Switch index {index} has no connected input. Connected: {', '.join(sorted(options))}")
        return io.NodeOutput(options[key])


class SystemOneExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SystemOneQuestion,
            SystemOne,
            SystemOneAnswer,
            SystemOneThresholdSwitch,
            SystemOneChoiceSwitch,
        ]


async def comfy_entrypoint() -> SystemOneExtension:
    return SystemOneExtension()
