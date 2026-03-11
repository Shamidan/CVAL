from random import randint, random

from src.core.al._tf import al
from src.core.models import FramePrediction, BBoxScores, DetectionSamplingOnPremise

# TODO: преобразовать в нормальные тесты

frames_predictions = list(
        map(
            lambda x: FramePrediction(
                frame_id=f"{randint(0, 100)}",
                predictions=list(map(lambda _: BBoxScores(category_id=randint(0, 2), score=random(), probabilities=[0.3, 0.3, 1-0.6]), range(100)))
            ),
            range(100)
        )
    )

request = DetectionSamplingOnPremise(
        num_of_samples=200,
        bbox_selection_policy='sum',
        selection_strategy='margin',
        sort_strategy='ascending',
        frames=frames_predictions,
        probs_weights=[1, 10, 20]
)

print(al(request))