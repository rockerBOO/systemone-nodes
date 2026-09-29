import json

import pytest

from systemone_nodes.nodes import (
    SystemOne,
    SystemOneAnswer,
    SystemOneChoiceSwitch,
    SystemOneQuestion,
    SystemOneThresholdSwitch,
    parse_criteria,
)


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


def test_choice_duplicate_key_raises():
    with pytest.raises(ValueError, match="'anime'"):
        parse_criteria("choice", "anime: Anime\nanime: Also anime")


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


def answer(name, question, response_answer):
    return SystemOneAnswer.execute({name: {"question": question, "answer": response_answer}}, name).result


def test_choice_answer():
    question = {"type": "choice", "instructions": "Team", "criteria": {"billing": "b", "technical": "t", "sales": "s"}}
    response = {"type": "choice", "choice": "technical", "probabilities": {"technical": 0.85, "billing": 0.15}, "confidence": 0.78}

    assert answer("department", question, response) == ("technical", 1, 0.85, 0.78, [0.15, 0.85, 0.0])


def test_score_answer():
    question = {"type": "score", "instructions": "Frustration", "criteria": ["Calm", "Frustrated", "Very angry"]}
    response = {
        "type": "score",
        "score": 1.4,
        "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
        "probabilities": {"0": 0.0, "1": 0.6, "2": 0.4},
        "confidence": 0.5,
    }

    assert answer("frustration", question, response) == ("1", 1, 1.4, 0.5, [0.0, 0.6, 0.4])


def test_noul_answer_yes():
    choice, index, value, confidence, probabilities = answer("refund", {"type": "noul", "instructions": "Refund?"}, {"type": "noul", "noul": 0.91})

    assert (choice, index, value, confidence) == ("true", 1, 0.91, 1.0)
    assert probabilities == pytest.approx([0.09, 0.91])


def test_noul_answer_no():
    choice, index, *_ = answer("refund", {"type": "noul", "instructions": "Refund?"}, {"type": "noul", "noul": 0.3})

    assert (choice, index) == ("false", 0)


def test_score_answer_rounds_half_up():
    question = {"type": "score", "instructions": "Frustration", "criteria": ["Calm", "Frustrated", "Very angry", "Livid"]}

    choice, index, *_ = answer(
        "frustration",
        question,
        {"type": "score", "score": 2.5, "probabilities": {"0": 0.0, "1": 0.0, "2": 0.5, "3": 0.5}, "confidence": 0.5},
    )
    assert (choice, index) == ("3", 3)

    choice, index, *_ = answer(
        "frustration",
        question,
        {"type": "score", "score": 0.5, "probabilities": {"0": 0.5, "1": 0.5, "2": 0.0, "3": 0.0}, "confidence": 0.5},
    )
    assert (choice, index) == ("1", 1)


def test_unsupported_answer_type_raises():
    with pytest.raises(ValueError, match="'oops'"):
        answer("x", {"type": "oops", "instructions": "?"}, {"type": "oops"})


def test_unknown_answer_name_lists_available():
    answers = {"route": {"question": ROUTE[1], "answer": EXAMPLE_RESPONSE["answers"]["route"]}}

    with pytest.raises(ValueError, match="Available: route"):
        SystemOneAnswer.execute(answers, "angry")


def test_threshold_switch_requests_selected_branch():
    assert SystemOneThresholdSwitch.check_lazy_status(0.9, 0.8, on_true=None, on_false=None) == ["on_true"]
    assert SystemOneThresholdSwitch.check_lazy_status(0.5, 0.8, on_true=None, on_false=None) == ["on_false"]


def test_threshold_switch_value_equal_to_threshold_is_true():
    assert SystemOneThresholdSwitch.execute(0.8, 0.8, on_true="yes", on_false="no").result == ("yes",)


def test_threshold_switch_below_threshold_is_false():
    assert SystemOneThresholdSwitch.execute(0.2, 0.8, on_true="yes", on_false="no").result == ("no",)


def test_threshold_switch_unconnected_branch_outputs_none():
    assert SystemOneThresholdSwitch.execute(0.9, 0.8, on_false="no").result == (None,)


def test_choice_switch_requests_only_selected_option():
    assert SystemOneChoiceSwitch.check_lazy_status(1, option0=None, option1=None) == ["option1"]


def test_choice_switch_outputs_selected_option():
    assert SystemOneChoiceSwitch.execute(1, option0=None, option1="b").result == ("b",)


def test_choice_switch_unconnected_index_raises():
    with pytest.raises(ValueError, match="index 3.*option0, option1"):
        SystemOneChoiceSwitch.execute(3, option0="a", option1="b")
