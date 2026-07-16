import pytest
from tests._fakes import RecordingGenerator
from eval.llm_judge import holistic_judge, JudgeOutput

def test_majority_vote_median():
    # Make a mock generator returning different scores
    class VotingFakeGenerator:
        def __init__(self):
            self.count = 0
            self.seeds = []
        def complete(self, messages, *, response_model=None, seed=None, **_):
            self.seeds.append(seed)
            # Return scores by call order (0.8, then 0.4, then 0.9) — independent
            # of the actual seed value, since callers may pass any base_seed.
            scores = [0.8, 0.4, 0.9]
            score = scores[self.count]
            self.count += 1
            return type("LLMResponse", (), {
                "parsed": {"score": score, "rationale": f"Vote {seed}"}
            })
            
    gen = VotingFakeGenerator()
    res = holistic_judge("Q", "A", ["C"], gen, votes=3, base_seed=10)
    assert gen.count == 3
    assert gen.seeds == [10, 11, 12]
    # Median of [0.4, 0.8, 0.9] is 0.8
    assert res["score"] == 0.8
    assert "Vote 10" in res["rationale"] or "Vote 12" in res["rationale"]  # corresponds to 0.8 or general context
