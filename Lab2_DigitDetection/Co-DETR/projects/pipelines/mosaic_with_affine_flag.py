import random

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines.transforms import Mosaic, RandomAffine


@PIPELINES.register_module()
class MosaicWithAffineFlag(Mosaic):
    """Mosaic that records whether the transform was actually applied."""

    def __init__(self, flag_key='_mosaic_applied', **kwargs):
        self.flag_key = flag_key
        super().__init__(**kwargs)

    def __call__(self, results):
        applied = random.uniform(0, 1) <= self.prob
        results[self.flag_key] = applied
        if not applied:
            return results

        results = self._mosaic_transform(results)
        results[self.flag_key] = True
        return results


@PIPELINES.register_module()
class ConditionalRandomAffine(RandomAffine):
    """RandomAffine that uses different borders for mosaic and plain samples."""

    def __init__(self,
                 mosaic_border=(0, 0),
                 plain_border=(0, 0),
                 flag_key='_mosaic_applied',
                 skip_plain_affine=False,
                 **kwargs):
        self.mosaic_border = mosaic_border
        self.plain_border = plain_border
        self.flag_key = flag_key
        self.skip_plain_affine = skip_plain_affine
        super().__init__(border=mosaic_border, **kwargs)

    def __call__(self, results):
        mosaic_applied = bool(results.pop(self.flag_key, False))
        if self.skip_plain_affine and not mosaic_applied:
            return results

        original_border = self.border
        self.border = self.mosaic_border if mosaic_applied else self.plain_border
        try:
            return super().__call__(results)
        finally:
            self.border = original_border
