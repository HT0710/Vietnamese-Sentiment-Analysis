from lightning.pytorch import Trainer, LightningModule
import lightning.pytorch.callbacks as cb
from rich import print


class TrainerCallback(cb.Callback):
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        epoch = trainer.current_epoch
        results = trainer.callback_metrics
        result_train = ', '.join([
            f"loss: {results['train/loss']:.4f}",
            f"acc: {results['train/accuracy']:.3f}",
            f"f1: {results['train/f1_score']:.3f}"
        ])
        result_val = ', '.join([
            f"loss: {results['val/loss']:.4f}",
            f"acc: {results['val/accuracy']:.3f}",
            f"f1: {results['val/f1_score']:.3f}",
        ])
        print(f"[bold]Epoch[/]( {epoch} )  [bold]Train[/]({result_train})  [bold]Val[/]({result_val})")


CALLBACKS = [
    cb.RichModelSummary(),
    cb.RichProgressBar(),
    cb.LearningRateMonitor(logging_interval='epoch'),
    cb.ModelCheckpoint(monitor='val/loss', save_weights_only=True),
    cb.EarlyStopping(monitor='val/loss', min_delta=0.001, patience=2),
    TrainerCallback(),
]
