from typing import List
import torch
from poseResearch.visualizer.pose_3d_visualizer import Pose3DVisualizer
from quantization.fast_quantization import FASTQuantizer
import json
from prediction.data.preparation import load_json, process_data
import logging
from prediction.model import GPT
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Default: log everything; filter with handlers

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # Default: show INFO and above

# Format including file name
formatter = logging.Formatter('%(levelname)s | %(filename)s:%(lineno)d | %(message)s')
console_handler.setFormatter(formatter)

# Add the handler
logger.addHandler(console_handler)

def test_pose_reconstruction(
    save: bool = True,
    visualize: bool = False,
    lifted_data: str = "poseResearch/dataloader/male2_t2_cam01",
):
    data = load_json(lifted_data)
    data = [d for d in data if "poselifting" in d]
    ceil_new_tokens = 60
    quantizer = FASTQuantizer("poseResearch/prediction/data/tokenizer")
    processed = process_data(data)
    forwarded = quantizer.forward(processed)
    quantized = forwarded["encoded"]
    model, _ = GPT.from_checkpoint("out/ckpt.pt")
    all_poses = []
    mean_correct_tokens = (0., 0.)
    for i, quantized_chunk in enumerate(quantized):
        quantized_chunk.append(quantizer.tokenizer.bpe_tokenizer.eos_token_id)
        max_new_tokens = (
            len(quantized[i + 1]) + 1 if i < len(quantized) - 1 else ceil_new_tokens
        )

        # print(f"Quantized poses: {len(quantized_chunk)} output encodings")

        out = model.generate(
            torch.Tensor(quantized_chunk).unsqueeze(0).long(), max_new_tokens
        )  # Add batch dimension and convert to long
        logger.info(f"Generated output shape: {out.size()}")
        out: List[int] = out.squeeze().tolist()  # Convert to list for easier printing

        try:
            mc = (torch.tensor(quantized[i+1]) == torch.tensor(out[len(quantized_chunk):len(quantized_chunk) + len(quantized[i+1])])).float().mean().item()
            logger.info(
                f"Correct tokens: {mc}"
            )
            mean_correct_tokens = (
                mean_correct_tokens[0] + mc*len(quantized[i+1]),
                mean_correct_tokens[1] + len(quantized[i+1])
            )
        except Exception as e:
            print(f"Error calculating correct tokens: {e}")
        # print(f"Output: {out}")

        # pre_gen_decoded = quantizer.decode([quantized_chunk])
        # print(f"Decoded shape: {pre_gen_decoded.shape}")
        # print(f"Decoded: {pre_gen_decoded[0]}")
        new_tokens = out[len(quantized_chunk):]
        split_tokens = []
        current_chunk = []
        for token in new_tokens:
            if token == quantizer.tokenizer.bpe_tokenizer.eos_token_id:
                split_tokens.append(current_chunk)
                current_chunk = []
            else:
                current_chunk.append(token)
        if current_chunk:
            split_tokens.append(current_chunk)

        for j, chunk in enumerate(split_tokens):
            unshaped_decoded = quantizer.decode( [chunk])
            if unshaped_decoded is None:
                logger.error(f"Error decoding chunk {j} in iteration {i}")
                continue
            try:
                quantized_decoded = quantizer.shape_back(unshaped_decoded, i)
            except Exception as e:
                logger.error(f"Error reshaping decoded chunk {j} in iteration {i}: {e}")
                continue
            all_poses.append(quantized_decoded)
            if visualize:
                visualize_3d_poses = Pose3DVisualizer(
                    skeleton_type="anatomical",
                    output_dir="./pose_video_output_3d",
                    create_videos=True,
                    video_fps=30,
                    save_plots=False,
                )
                try:
                    visualize_3d_poses.visualize_3d_poses(
                        quantized_decoded.unsqueeze(0), {"batch_idx": 0}, "poselifting"
                    )
                    logger.info(f"Decoded pose {j}: {quantized_decoded}")
                    logger.info(f"Decoded shape: {quantized_decoded.shape}")
                except Exception as e:
                    logger.error(f"Error visualizing pose {j}: {e}")
                    break

    logger.info(f"Mean correct tokens: {mean_correct_tokens[0] / mean_correct_tokens[1] if mean_correct_tokens[1] > 0 else 0}")
    if save:
        # Convert all poses to a list and create the required format
        poses_list = torch.cat(all_poses).unsqueeze(0)
        logger.info(f"Total poses reconstructed: {poses_list.size()}")
        output_data = {"poselifting": {"data": poses_list.tolist()}}

        # Save to JSON file
        with open("reconstructed_poses.json", "w") as f:
            json.dump(output_data, f)


if __name__ == "__main__":
    test_pose_reconstruction()
    print("Pose reconstruction test completed successfully.")
