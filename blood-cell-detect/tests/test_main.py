def test_hello_world(monkeypatch):
    import sys
    from src.main import hello_world

    # Capture the output of the hello_world function
    monkeypatch.setattr(sys, 'stdout', open('output.txt', 'w'))
    hello_world()
    with open('output.txt', 'r') as f:
        output = f.read().strip()

    assert output == "Hello, World!"