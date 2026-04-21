from mmcv.parallel import is_module_wrapper
from mmcv.runner.hooks import HOOKS, Hook


@HOOKS.register_module()
class MultiImageMixAugSwitchHook(Hook):
    """Disable selected multi-image-mix transforms in the last epochs.

    Unlike YOLOXModeSwitchHook, this hook only updates the dataset wrapper's
    skip list and does not touch any model-specific attributes such as
    ``bbox_head.use_l1``.
    """

    def __init__(self, num_last_epochs=6, skip_type_keys=('Mosaic', 'MixUpWithProb')):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._restart_dataloader = False

    def before_train_epoch(self, runner):
        epoch = runner.epoch
        train_loader = runner.data_loader
        model = runner.model
        if is_module_wrapper(model):
            model = model.module

        if (epoch + 1) == runner.max_epochs - self.num_last_epochs:
            runner.logger.info('Disable selected multi-image-mix aug in final epochs.')
            if hasattr(train_loader.dataset, 'update_skip_type_keys'):
                train_loader.dataset.update_skip_type_keys(self.skip_type_keys)
            else:
                runner.logger.warning(
                    'Train dataset does not support update_skip_type_keys; '
                    'skipping late-stage augmentation switch.'
                )
                return

            if getattr(train_loader, 'persistent_workers', False):
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
        elif self._restart_dataloader:
            train_loader._DataLoader__initialized = True
