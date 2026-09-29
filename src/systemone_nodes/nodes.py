import os

import requests
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io

Question = io.Custom("SYSTEMONE_QUESTION")
Answers = io.Custom("SYSTEMONE_ANSWERS")


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


class SystemOneExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            SystemOneQuestion,
            SystemOne,
        ]


async def comfy_entrypoint() -> SystemOneExtension:
    return SystemOneExtension()
