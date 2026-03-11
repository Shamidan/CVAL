import uuid
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field, validator


class EnumOnPremiseBBoxSelectionPolicy(str, Enum):
    min, max, sum, mean = 'min', 'max', 'sum', 'mean'


class EnumOnPremiseBBoxSelectionStrategy(str, Enum):
    margin = 'margin'
    least = 'least'
    ratio = 'ratio'
    entropy = 'entropy'
    probability = 'probability'
    cval_2_p = 'cval_custom'


class EnumOnPremiseBBoxSortStrategy(str, Enum):
    ascending, descending = 'ascending', 'descending'


class BBoxScores(BaseModel):
    category_id: Optional[int] = Field(
        description='id of the predicted class',
        example=0,
    )
    score: Optional[float] = Field(
        description='prediction score. Real value from 0 to 1',
    )
    probabilities: Optional[List[float]] = Field(
        description=(
            'the probabilities for each object category are relative to a predicted bounding box.'
            ' The order in the list is determined by the category number. sum must be = 1'
        )
    )

    @validator('score')
    def validate_score(cls, value):
        if value is not None and not (0 < value < 1):
            raise ValueError('the predicted score should be in the range (0, 1)')
        return value

    @validator('probabilities')
    def validate_probabilities(cls, value: Optional[List[float]]):
        if value is not None:
            for prob in value:
                if prob < 0:
                    raise ValueError('Each probability must be > 0')
        return value


class FramePrediction(BaseModel):
    frame_id: str = Field(description='id of the frame', example=uuid.uuid4())
    predictions: List[BBoxScores] = Field(description='bbox_scores', )


class DetectionSamplingOnPremise(BaseModel):
    use_null_detections: bool = Field(default=False)
    num_of_samples: int = 100
    bbox_selection_policy: Optional[EnumOnPremiseBBoxSelectionPolicy] = Field(
        description=(
            'which bounding box to select when there are multiple boxes on an image, '
            'according to their confidence. Currently supports: min, max, sum, mean'
        ),
        example='mean',
    )
    sort_strategy: Optional[EnumOnPremiseBBoxSortStrategy] = Field(
        description='Sorting strategy. Currently supports: ascending, descending',
        example='ascending',
    )
    probs_weights: Optional[list[int]] = Field(
        description=(
            'Determines the significance (weight) of the prediction probability for each class. '
            'The order in the list corresponds to the order of the classes.'
            ' It is essential for a multi-class entropy method.'
        ),
    )
    selection_strategy: EnumOnPremiseBBoxSelectionStrategy = Field(
        description='selection strategy. Currently supports: margin, least, ratio, entropy',
        example='entropy',
    )

    frames: List[FramePrediction]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.val_bbox_policy_sort_strategy()
        self.val_dataset_id()
        self.val_prob_weights()
        self.val_null_detections()

    def val_bbox_policy_sort_strategy(self):
        if self.selection_strategy not in (
                'clustering',
                'hierarchical',
        ) and None in (self.sort_strategy, self.bbox_selection_policy):
            raise ValueError(
                f'check request params: '
                f'{"bbox_selection_policy " if self.bbox_selection_policy is None else ""}'
                f'{"sort_strategy" if self.sort_strategy is None else ""}'.lstrip(' ')
            )

    def val_prob_weights(self):
        if self.probs_weights:
            for prediction in map(lambda x: x.predictions, self.frames):
                for probs in prediction:
                    if probs.probabilities is not None:
                        if len(probs.probabilities) != len(self.probs_weights):
                            raise ValueError('length of probabilities must be equal to length of probs_weights')

    def val_null_detections(self):
        if not self.use_null_detections and not self.probs_weights:
            for frame in self.frames:
                if tuple(filter(lambda x: None in (x.score, x.category_id), frame.predictions)):
                    raise ValueError('use_null_detections disabled, mc_task_id not passed')

    def val_dataset_id(self):
        if self.selection_strategy in ('clustering', 'hierarchical') and self.dataset_id is None:
            raise ValueError('passed null dataset_id for clustering. must be not null')
