from typing import List
import torch
from poseResearch.visualizer.pose_3d_visualizer import Pose3DVisualizer
from quantization.fast_quantization import FASTQuantizer
import json
from prediction.data.preparation import load_json, process_data
from prediction.model import GPT


def test_pose_reconstruction(save: bool = True, visualize: bool = False, lifted_data: str = "poseResearch/dataloader"):
    data = load_json(lifted_data)
    max_new_tokens = 60
    quantizer = FASTQuantizer("poseResearch/prediction/data/tokenizer")
    processed = process_data(data)
    quantized = quantizer.quantize(processed)
    model, _ = GPT.from_checkpoint("out/ckpt.pt")
    all_poses = []
    for i, quantized_chunk in enumerate(quantized):
        quantized_chunk.append(2048)
        max_new_tokens = len(quantized[i+1]) + 1 if i < len(quantized) - 1 else max_new_tokens

        #print(f"Quantized poses: {len(quantized_chunk)} output encodings")


        out = model.generate(torch.Tensor(quantized_chunk).unsqueeze(0).long(), max_new_tokens)  # Add batch dimension and convert to long
        #print(f"Generated output shape: {out.size()}")
        out: List[int] = out.squeeze().tolist()  # Convert to list for easier printing
        try:
            print(f"Correct tokens: {(torch.tensor(quantized[i+1]) == torch.tensor(out[len(quantized_chunk):len(quantized_chunk) + len(quantized[i+1])])).float().mean()}")
        except Exception as e:
            print(f"Error calculating correct tokens: {e}")
        #print(f"Output: {out}")

        #pre_gen_decoded = quantizer.decode([quantized_chunk])
        #print(f"Decoded shape: {pre_gen_decoded.shape}")
        #print(f"Decoded: {pre_gen_decoded[0]}")
        new_tokens = out[len(quantized_chunk):]
        split_tokens = []
        current_chunk = []
        for token in new_tokens:
            if token == 2048:
                split_tokens.append(current_chunk)
                current_chunk = []
            else:
                current_chunk.append(token)
        if current_chunk:
            split_tokens.append(current_chunk)
        

            
        for j, chunk in enumerate(split_tokens):
            quantized_decoded = quantizer.shape_back(quantizer.decode([chunk]))
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
                    visualize_3d_poses.visualize_3d_poses(quantized_decoded.unsqueeze(0), {"batch_idx":0}, "poselifting")
                    print(f"Decoded pose {j}: {quantized_decoded}")
                    print(f"Decoded shape: {quantized_decoded.shape}")
                except Exception as e:
                    print(f"Error visualizing pose {j}: {e}")
                    break

    if save: 
    # Convert all poses to a list and create the required format
        poses_list = [pose.tolist()[0] for pose in all_poses if not (pose == 0.0).all()]
        output_data = {
            "poselifting": {
            "data": [poses_list]
            }
        }
        
        # Save to JSON file
        with open('reconstructed_poses.json', 'w') as f:
            json.dump(output_data, f)





if __name__ == "__main__":
    test_pose_reconstruction()
    print("Pose reconstruction test completed successfully.")