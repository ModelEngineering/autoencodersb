'''Runs an ensemble of model runners.'''

import iplane.constants as cn  # type: ignore
from iplane.model_runner import ModelRunner, RunnerResult  # type: ignore

from collections import namedtuple
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import torch
from typing import List, Union, cast


ModelEnsemble = namedtuple('ModelEnsemble', ['models'])


class ModelRunnerEnsemble(ModelRunner):
    """Ensemble of model runners."""

    def __init__(self, model_runners: List[ModelRunner], **kwargs):
        """
        Args:
            model_runners (List[ModelRunner]): List of model runners to ensemble.
        """
        models = [runner.model for runner in model_runners]
        super().__init__(model=ModelEnsemble(models=models))
        self.model_runners = model_runners

    def fit(self, train_dl: torch.utils.data.DataLoader) -> RunnerResult:
        """Fit each model runner in the ensemble."""
        losses = []
        for runner in self.model_runners:
            result = runner.fit(train_dl)
            losses.extend(result.losses)
        avg_loss = np.mean([loss.item() for loss in losses])
        return RunnerResult(avg_loss=avg_loss, losses=losses)

    def predict(self, features: Union[np.ndarray, pd.DataFrame]) -> torch.Tensor:
        """Predict using the ensemble of model runners."""
        predictions = [runner.predict(features) for runner in self.model_runners]
        return torch.mean(torch.stack(predictions), dim=0)