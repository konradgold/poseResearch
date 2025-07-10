import torch
from quantization.fast_quantization import FASTQuantizer
import json


def check_fast_quantizer(lifted_data: str = "poseResearch/dataloader/results_3d.json"):
    # pm = ProcessManager() # do not want to store anything
    fast_quantizer = FASTQuantizer()

    with open(lifted_data, "r") as file:
        data = json.loads(file.read())

    poses = torch.Tensor(data["poselifting"]["data"]).squeeze()
    assert len(poses.size()) == 3
    assert poses.size(-2) == 17
    assert poses.size(-1) == 3

    assert poses.size(0) > 8

    fast_quantizer.fit_tokenizer(poses[: 3 * len(poses) // 4, :])

    out = fast_quantizer.forward(poses[3 * len(poses) // 4 :, :])
    print(f"Loss: {out['loss']}")
    print(f"Tokenized input: {out['encoded']}")

    print("Checked Quantizer")


def check_transformer_quantizer(
    lifted_data: str = "poseResearch/dataloader/results_3d.json",
): ...


if __name__ == "__main__":
    check_fast_quantizer()
