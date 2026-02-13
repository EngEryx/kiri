"""Pipe — linear composition engine for chaining atoms and functions."""

import time


class Pipe:
    """
    Connects atoms. Output of one atom's prediction feeds into
    another atom's input. This is how molecules form.

    Each stage is a function that takes input and returns output.
    An atom's predict_next IS a pipe stage.
    """

    def __init__(self):
        self.stages = []
        self.log = []

    def add(self, name, fn):
        """Add a stage. fn: takes input, returns output."""
        self.stages.append((name, fn))
        return self

    def run(self, initial_input):
        """Run the pipeline. Each stage's output feeds the next."""
        data = initial_input
        for name, fn in self.stages:
            data = fn(data)
            self.log.append({
                'stage': name,
                'output': str(data)[:200],
                'time': time.time()
            })
        return data

    def loop(self, initial_input, iterations=1, delay=0):
        """Run the pipeline in a loop, feeding output back to input."""
        data = initial_input
        for i in range(iterations):
            data = self.run(data)
            if delay > 0:
                time.sleep(delay)
        return data
