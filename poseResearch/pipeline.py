import torch
from estimation.utils import Estimation
from utils.state_saver import StateSaver

class EstimationPipe:
    def __init__(self, preprocessor: Estimation, flatpose: Estimation, poselifting: Estimation, state_saver: StateSaver):
        self.pipe_classes = [preprocessor, flatpose, poselifting]
        self.state_saver = state_saver

    
    def forward(self, dataloader):
        for batch in dataloader:
            for module in self.pipe_classes:
                state = module.forward(batch) # shape (#persons in batch, #frames, 17,3)
                self.state_saver.handle(state, module.config)
            assert isinstance(state, torch.Tensor)
            assert state.size(1) == batch.size(0)
            assert state.size(2) == 17
            assert state.size(3) == 3
            yield state
        