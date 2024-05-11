# The model.

# >>>
if __name__ == '__main__':
    import os
    import sys

    _project_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    os.environ['PROJECT_DIR'] = _project_dir
    sys.path.append(_project_dir)
    del _project_dir
# <<<

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Union

import delu
import faiss
import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.tensorboard
from loguru import logger
from torch import Tensor
from tqdm import tqdm

import lib
from lib import KWArgs

# dis-entangled models
from .tabr_components import Encoder, RetrieveModule


@dataclass(frozen=True)
class Config:
    seed: int
    data: Union[lib.Dataset[np.ndarray], KWArgs]  # lib.data.build_dataset
    model: KWArgs  # Model
    context_size: int
    optimizer: KWArgs  # lib.deep.make_optimizer
    batch_size: int
    patience: Optional[int]
    n_epochs: Union[int, float]
    
class Model(nn.Module):
    def __init__(
        self,
        *,
        #
        n_num_features: int,
        n_bin_features: int,
        cat_cardinalities: list[int],
        n_classes: Optional[int],
        #
        num_embeddings: Optional[dict],  # lib.deep.ModuleSpec
        d_main: int,
        d_multiplier: float,
        encoder_n_blocks: int,
        predictor_n_blocks: int,
        mixer_normalization: Union[bool, Literal['auto']],
        context_dropout: float,
        dropout0: float,
        dropout1: Union[float, Literal['dropout0']],
        normalization: str,
        activation: str,
        #
        # The following options should be used only when truly needed.
        memory_efficient: bool = False,
        candidate_encoding_batch_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            n_num_features=n_num_features,
            n_bin_features=n_bin_features,
            cat_cardinalities=cat_cardinalities,
            num_embeddings=num_embeddings,
            d_main=d_main,
            d_multiplier=d_multiplier,
            encoder_n_blocks=encoder_n_blocks,
            mixer_normalization=mixer_normalization,
            dropout0=dropout0,
            dropout1=dropout1,
            normalization=normalization,
            activation=activation
        )
        
        self.retrieve_module = RetrieveModule(
            encoder=self.encoder,
            n_classes=n_classes,
            context_dropout=context_dropout,
            memory_efficient=memory_efficient,
            candidate_encoding_batch_size=candidate_encoding_batch_size
        )
        
        Normalization = getattr(nn, normalization)
        
        def make_block(prenorm: bool) -> nn.Sequential:
            return nn.Sequential(
                *([Normalization(d_main)] if prenorm else []),
                nn.Linear(d_main, self.encoder.d_block),
                self.encoder.activation,
                nn.Dropout(dropout0),
                nn.Linear(self.encoder.d_block, d_main),
                nn.Dropout(dropout1),
            )
        
        self.blocks1 = nn.ModuleList(
            [make_block(True) for _ in range(predictor_n_blocks)]
        )
        
        self.head = nn.Sequential(
            Normalization(d_main),
            self.encoder.activation,
            nn.Linear(d_main, lib.get_d_out(n_classes)),
        )
        
    def forward(
        self,
        *,
        x_: dict[str, Tensor], #* BATCH of input data
        y: Optional[Tensor],
        candidate_x_: dict[str, Tensor],
        candidate_y: Tensor,
        context_size: int,
        is_train: bool,
    ) -> Tensor:
        
        x = self.retrieve_module(
            x_, y, candidate_x_, candidate_y, context_size, is_train
        )
        
        # logger.info(f"Retrieve Module Output: {type(x)}, shape: {x.shape}")
        
        for block in self.blocks1:
            x = x + block(x)
        x = self.head(x)
        
        # logger.info(f"Model Output: {type(x)}, shape: {x.shape}")
        
        return x
        

def main(
    config: lib.JSONDict, output: Union[str, Path], *, force: bool = False
) -> Optional[lib.JSONDict]:
    # >>> start
    if not lib.start(output, force=force):
        return None

    output = Path(output)
    report = lib.create_report(config)
    C = lib.make_config(Config, config)

    delu.random.seed(C.seed)
    device = lib.get_device()

    # >>> data
    dataset = (
        C.data if isinstance(C.data, lib.Dataset) else lib.build_dataset(**C.data)
    ).to_torch(device)
    if dataset.is_regression:
        dataset.data['Y'] = {k: v.float() for k, v in dataset.Y.items()}
    Y_train = dataset.Y['train'].to(
        torch.long if dataset.is_multiclass else torch.float
    )

    # >>> model
    model = Model(
        n_num_features=dataset.n_num_features,
        n_bin_features=dataset.n_bin_features,
        cat_cardinalities=dataset.cat_cardinalities(),
        n_classes=dataset.n_classes(),
        **C.model,
    )
    report['n_parameters'] = lib.get_n_parameters(model)
    logger.info(f'n_parameters = {report["n_parameters"]}')
    report['prediction_type'] = None if dataset.is_regression else 'logits'
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # type: ignore[code]

    # >>> training
    def zero_wd_condition(
        module_name: str,
        module: nn.Module,
        parameter_name: str,
        parameter: nn.parameter.Parameter,
    ):
        return (
            'label_encoder' in module_name
            or 'label_encoder' in parameter_name
            or lib.default_zero_weight_decay_condition(
                module_name, module, parameter_name, parameter
            )
        )

    optimizer = lib.make_optimizer(
        model, **C.optimizer, zero_weight_decay_condition=zero_wd_condition
    )
    loss_fn = lib.get_loss_fn(dataset.task_type)

    train_size = dataset.size('train')
    train_indices = torch.arange(train_size, device=device)

    epoch = 0
    eval_batch_size = 32768
    chunk_size = None
    progress = delu.ProgressTracker(C.patience)
    training_log = []
    writer = torch.utils.tensorboard.SummaryWriter(output)  # type: ignore[code]

    def get_Xy(part: str, idx) -> tuple[dict[str, Tensor], Tensor]:
        batch = (
            {
                key[2:]: dataset.data[key][part]
                for key in dataset.data
                if key.startswith('X_')
            },
            dataset.Y[part],
        )
        return (
            batch
            if idx is None
            else ({k: v[idx] for k, v in batch[0].items()}, batch[1][idx])
        )

    def apply_model(part: str, idx: Tensor, training: bool):
        x, y = get_Xy(part, idx)

        candidate_indices = train_indices
        is_train = part == 'train'
        if is_train:
            # NOTE: here, the training batch is removed from the candidates.
            # It will be added back inside the model's forward pass.
            candidate_indices = candidate_indices[~torch.isin(candidate_indices, idx)]
        candidate_x, candidate_y = get_Xy(
            'train',
            # This condition is here for historical reasons, it could be just
            # the unconditional `candidate_indices`.
            None if candidate_indices is train_indices else candidate_indices,
        )

        return model(
            x_=x,
            y=y if is_train else None,
            candidate_x_=candidate_x,
            candidate_y=candidate_y,
            context_size=C.context_size,
            is_train=is_train,
        ).squeeze(-1)

    @torch.inference_mode()
    def evaluate(parts: list[str], eval_batch_size: int):
        model.eval()
        predictions = {}
        for part in parts:
            while eval_batch_size:
                try:
                    predictions[part] = (
                        torch.cat(
                            [
                                apply_model(part, idx, False)
                                for idx in torch.arange(
                                    dataset.size(part), device=device
                                ).split(eval_batch_size)
                            ]
                        )
                        .cpu()
                        .numpy()
                    )
                except RuntimeError as err:
                    if not lib.is_oom_exception(err):
                        raise
                    eval_batch_size //= 2
                    logger.warning(f'eval_batch_size = {eval_batch_size}')
                else:
                    break
            if not eval_batch_size:
                RuntimeError('Not enough memory even for eval_batch_size=1')
        metrics = (
            dataset.calculate_metrics(predictions, report['prediction_type'])
            if lib.are_valid_predictions(predictions)
            else {x: {'score': -999999.0} for x in predictions}
        )
        return metrics, predictions, eval_batch_size

    def save_checkpoint():
        lib.dump_checkpoint(
            {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'random_state': delu.random.get_state(),
                'progress': progress,
                'report': report,
                'timer': timer,
                'training_log': training_log,
            },
            output,
        )
        lib.dump_report(report, output)
        lib.backup_output(output)

    print()
    timer = lib.run_timer()
    while epoch < C.n_epochs:
        print(f'[...] {lib.try_get_relative_path(output)} | {timer}')

        model.train()
        epoch_losses = []
        for batch_idx in tqdm(
            lib.make_random_batches(train_size, C.batch_size, device),
            desc=f'Epoch {epoch}',
        ):
            loss, new_chunk_size = lib.train_step(
                optimizer,
                lambda idx: loss_fn(apply_model('train', idx, True), Y_train[idx]),
                batch_idx,
                chunk_size or C.batch_size,
            )
            epoch_losses.append(loss.detach())
            if new_chunk_size and new_chunk_size < (chunk_size or C.batch_size):
                chunk_size = new_chunk_size
                logger.warning(f'chunk_size = {chunk_size}')

        epoch_losses, mean_loss = lib.process_epoch_losses(epoch_losses)
        metrics, predictions, eval_batch_size = evaluate(
            ['val', 'test'], eval_batch_size
        )
        lib.print_metrics(mean_loss, metrics)
        training_log.append(
            {'epoch-losses': epoch_losses, 'metrics': metrics, 'time': timer()}
        )
        writer.add_scalars('loss', {'train': mean_loss}, epoch, timer())
        for part in metrics:
            writer.add_scalars('score', {part: metrics[part]['score']}, epoch, timer())

        progress.update(metrics['val']['score'])
        if progress.success:
            lib.celebrate()
            report['best_epoch'] = epoch
            report['metrics'] = metrics
            save_checkpoint()
            lib.dump_predictions(predictions, output)

        elif progress.fail or not lib.are_valid_predictions(predictions):
            break

        epoch += 1
        print()
    report['time'] = str(timer)

    # >>> finish
    model.load_state_dict(lib.load_checkpoint(output)['model'])
    report['metrics'], predictions, _ = evaluate(
        ['train', 'val', 'test'], eval_batch_size
    )
    report['chunk_size'] = chunk_size
    report['eval_batch_size'] = eval_batch_size
    lib.dump_predictions(predictions, output)
    lib.dump_summary(lib.summarize(report), output)
    save_checkpoint()
    lib.finish(output, report)
    return report


if __name__ == '__main__':
    lib.configure_libraries()
    lib.run_Function_cli(main)
