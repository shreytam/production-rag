from tests._fakes import RecordingGenerator

def test_generator_receives_seed():
    gen = RecordingGenerator()
    # Explicitly verify we can invoke complete with seed
    gen.complete([], seed=42)
    assert len(gen.calls) == 1
    # Check that seed was stored/passed
    assert getattr(gen, "last_seed", None) == 42
