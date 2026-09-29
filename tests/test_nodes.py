import json

import pytest

from systemone_nodes.nodes import SystemOne, SystemOneQuestion, parse_criteria


def test_choice_criteria_keep_line_order():
    text = "anime: Anime or illustrated\n\nphoto: Photorealistic: shot on a camera\n"

    assert list(parse_criteria("choice", text).items()) == [
        ("anime", "Anime or illustrated"),
        ("photo", "Photorealistic: shot on a camera"),
    ]


def test_choice_line_without_colon_raises():
    with pytest.raises(ValueError, match="'photo'"):
        parse_criteria("choice", "anime: Anime\nphoto")


def test_score_criteria_is_ordered_list():
    assert parse_criteria("score", "Calm\n\n Frustrated \nVery angry") == ["Calm", "Frustrated", "Very angry"]


def test_question_output():
    name, question = SystemOneQuestion.execute("route", "choice", "Which team?", "1: Refunds\n2: Damaged").result[0]

    assert name == "route"
    assert question == {"type": "choice", "instructions": "Which team?", "criteria": {"1": "Refunds", "2": "Damaged"}}


def test_noul_without_criteria_omits_criteria():
    _, question = SystemOneQuestion.execute("angry", "noul", "Is the customer angry?", "").result[0]

    assert question == {"type": "noul", "instructions": "Is the customer angry?"}


ROUTE = (
    "route",
    {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"1": "Refunds and payments", "2": "Damaged or lost parcels", "3": "Account and login problems"},
    },
)
ANGRY = ("angry", {"type": "noul", "instructions": "Is the customer angry?"})
STATE = "Refund request: the customer says the parcel arrived crushed and wants their money back."
URL = "http://localhost:8765/v1/systemone"
EXAMPLE_RESPONSE = {
    "model": "jeff-qwen3.5-0.8b",
    "answers": {
        "route": {
            "type": "choice",
            "probabilities": {"1": 0.06129659929244851, "2": 0.9359839293802058, "3": 0.002719471327345735},
            "choice": "2",
            "confidence": 0.9039758940703085,
        },
        "angry": {"type": "noul", "noul": 0.7385468107028845},
    },
    "usage": {"input_tokens": 222, "output_tokens": 0},
}


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = json.dumps(body)
        self._body = body

    def json(self):
        return self._body


class FakePost:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


def install_post(monkeypatch, status_code=200, body=EXAMPLE_RESPONSE):
    post = FakePost(FakeResponse(status_code, body))
    monkeypatch.setattr("systemone_nodes.nodes.requests.post", post)
    return post


def test_request_body(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    post = install_post(monkeypatch)

    SystemOne.execute(URL, "jeff-latest", STATE, {"question0": ROUTE, "question1": ANGRY})

    url, kwargs = post.calls[0]
    assert url == URL
    assert kwargs["json"] == {
        "model": "jeff-latest",
        "state": STATE,
        "questions": {"route": ROUTE[1], "angry": ANGRY[1]},
    }
    assert kwargs["headers"] == {}
    assert kwargs["timeout"] == 60


def test_api_key_header_from_env(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "secret")
    post = install_post(monkeypatch)

    SystemOne.execute(URL, "jeff-latest", STATE, {"question0": ROUTE})

    assert post.calls[0][1]["headers"] == {"Authorization": "Bearer secret"}


def test_answers_pair_question_with_answer(monkeypatch):
    install_post(monkeypatch)

    answers = SystemOne.execute(URL, "jeff-latest", STATE, {"question0": ROUTE, "question1": ANGRY}).result[0]

    assert answers["route"]["question"] == ROUTE[1]
    assert answers["route"]["answer"]["choice"] == "2"
    assert answers["angry"]["answer"]["noul"] == 0.7385468107028845


def test_error_includes_status_and_body(monkeypatch):
    install_post(monkeypatch, status_code=422, body={"detail": "criteria required"})

    with pytest.raises(RuntimeError, match="422.*criteria required"):
        SystemOne.execute(URL, "jeff-latest", STATE, {"question0": ROUTE})


def test_duplicate_question_names_raise(monkeypatch):
    post = install_post(monkeypatch)

    with pytest.raises(ValueError, match="route"):
        SystemOne.execute(URL, "jeff-latest", STATE, {"question0": ROUTE, "question1": ROUTE})
    assert post.calls == []
