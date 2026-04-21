import random

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.transforms import MixUp


@PIPELINES.register_module()
class MixUpWithProb(MixUp):
    """MixUp with an explicit application probability."""

    def __init__(self, prob=1.0, **kwargs):
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f'prob must be in [0, 1], got {prob}')
        self.prob = float(prob)
        super().__init__(**kwargs)

    def __call__(self, results):
        if random.random() > self.prob:
            return results
        return super().__call__(results)
