import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def run_with_enforced_limits(time_limit=None):
    """
    Execute methods with resource limits. Works only for linux.
    """

    def signal_handler(signum, frame):
        raise TimeoutException("This job has timed out. The results will still be used")

    signal.signal(signal.SIGALRM, signal_handler)

    # Set alarm for time limit
    signal.alarm(time_limit)

    try:
        yield
    finally:
        signal.alarm(0)
