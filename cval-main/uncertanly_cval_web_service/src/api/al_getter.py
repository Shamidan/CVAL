from typing import Callable


class ALGetter:

    def __init__(self, al_func: Callable):
        self.al_func = al_func

    def __call__(self):
        return self.al_func
