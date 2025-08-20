import torch
from quantization.fast_quantization import FASTQuantizer
import json


def test_fast_quantizer(lifted_data: str = "/Volumes/KG1TB/Developement/poseResearch/poseResearch/dataloader/male2_t2_cam01/results_poselifting.json", iterations: int = 2, nr_tokens: int = 2000):
    # pm = ProcessManager() # do not want to store anything
    with open(lifted_data, "r") as file:
        data = json.loads(file.read())

    poses = torch.Tensor(data["poselifting"]["data"]).squeeze()
    assert len(poses.size()) == 3
    assert poses.size(-2) == 17
    assert poses.size(-1) == 3
    token_steps = [349, 360, 370, 380, 390, 400]
    
    print(poses[3 * len(poses) // 4 :, :].size())

    assert poses.size(0) > 8
    for j in token_steps:
        for i in range(iterations):
            try:
                print(f"Testing Quantizer with {j} tokens, next_joint_first={i==0}")
                fast_quantizer = FASTQuantizer()
                fast_quantizer.fit_tokenizer(poses[: 3 * len(poses) // 4, :], next_joint_first=i==0, num_tokens=j if nr_tokens > 0 else -1)
                out = fast_quantizer.forward(poses[3 * len(poses) // 4 :, :])


                
                print(f"Loss: {out['loss']}")
                print(f"Nr. Tokens required: {sum([len(l) for l in out["encoded"]])}")
                print(f"MPJPE: {out['mpjpe']}")

                print(f"Checked Quantizer iteration {i+1}/{iterations}")

            except:
                continue

if __name__ == "__main__":
    test_fast_quantizer(nr_tokens = 15000)
