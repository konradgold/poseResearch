import torch
from estimation.util import Estimation
from utils.output_saver import OutputSaver

class EstimationPipe:
    def __init__(self, preprocessor: Estimation, flatpose: Estimation, poselifting: Estimation, output_saver: OutputSaver):
        self.pipe_classes = [preprocessor, flatpose, poselifting]
        self.output_saver = output_saver

    
    def forward(self, dataloader):
        for batch in dataloader:
            for module in self.pipe_classes:
                output = module.forward(batch)
                self.output_saver.handle(output, module.config)
            assert isinstance(output, torch.Tensor)
            # shape (#persons in batch, #frames, 17,3)
            assert output.size(1) == batch.size(0)
            assert output.size(2) == 17
            assert output.size(3) == 3
            yield output
        