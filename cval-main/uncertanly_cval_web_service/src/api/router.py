from typing import List, Callable

from uncertanly_cval_web_service.src.core.depends.routable import BaseRoutable, post
from uncertanly_cval_web_service.src.core.models import DetectionSamplingOnPremise


class ALView(BaseRoutable):

    def __init__(self, al_func: Callable, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.al_func = al_func

    @post('al')
    def sampling(self, body: DetectionSamplingOnPremise) -> List[str]:
        return list(map(lambda x: x[0], self.al_func(body)))
