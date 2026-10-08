import time

import pytest

from forensic_rag.limits import OneAtATime, RateLimitError, SlidingWindow


def test_sliding_window_is_per_user_and_refills(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(time, "time", lambda: now[0])
    limit = SlidingWindow(2, 3600, "investigations")

    limit.hit("alice")
    limit.hit("alice")
    with pytest.raises(RateLimitError, match=r"2 investigations per 1 hour\. Try again in 1 hour"):
        limit.hit("alice")
    limit.hit("bob")                                  # other users are unaffected

    now[0] += 3601                                    # the oldest events leave the window
    limit.hit("alice")


def test_rejected_hits_are_not_recorded():
    limit = SlidingWindow(3, 60, "calls")
    limit.hit("alice", 2)
    with pytest.raises(RateLimitError):
        limit.hit("alice", 2)
    limit.hit("alice")                                # the failed batch of 2 did not count


def test_one_at_a_time():
    slots = OneAtATime("an investigation")
    with slots.slot("alice"):
        with pytest.raises(RateLimitError, match="already have an investigation running"):
            with slots.slot("alice"):
                pass
        with slots.slot("bob"):
            pass
    with slots.slot("alice"):                         # released after the first one finished
        pass


def test_slot_is_released_when_the_job_fails():
    slots = OneAtATime("an investigation")
    with pytest.raises(ValueError):
        with slots.slot("alice"):
            raise ValueError("boom")
    with slots.slot("alice"):
        pass
