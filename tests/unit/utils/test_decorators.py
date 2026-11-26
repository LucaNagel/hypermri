from hypermri.utils.system.decorators import onlycallonce


def test_onlycallonce_executes_default_decorated_method_once():
    class Worker:
        def __init__(self):
            self.calls = 0

        @onlycallonce
        def process(self, value):
            self.calls += 1
            return value * 2

    worker = Worker()

    assert worker.process(3) == 6
    assert worker.process(3) is None
    assert worker.calls == 1
    assert worker.__process_HasBeenCalledFlag is True


def test_onlycallonce_accepts_a_custom_flag_name():
    class Worker:
        @onlycallonce("completed")
        def process(self):
            return "done"

    worker = Worker()

    assert worker.process() == "done"
    assert worker.completed is True
    assert worker.process() is None
