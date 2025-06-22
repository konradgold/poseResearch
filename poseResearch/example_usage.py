"""
Example usage of the EstimationPipe with dedicated 2D and 3D visualizers
"""

import torch
from pipeline import EstimationPipe
from poseResearch.visualizer.pose_2d_visualizer import Pose2DVisualizer
from poseResearch.visualizer.pose_3d_visualizer import Pose3DVisualizer
# You would import your actual implementations here:
# from your_modules import YourPreprocessor, YourFlatpose, YourPoselifting, YourOutputSaver


def create_pipeline_with_dedicated_visualizers():
    """Example of pipeline with dedicated 2D and 3D visualizers (RECOMMENDED)"""
    
    # Create your actual estimation modules
    # preprocessor = YourPreprocessor()
    # flatpose = YourFlatpose() 
    # poselifting = YourPoselifting()
    # output_saver = YourOutputSaver()
    
    # Create dedicated visualizers
    visualizer_2d = Pose2DVisualizer(
        visualize_every_n_batches=2,  # Visualize 2D every 2nd batch
        save_plots=True,
        output_dir="./visualizations_2d_dedicated",
        show_labels=True,
        max_people=4
    )
    
    visualizer_3d = Pose3DVisualizer(
        visualize_every_n_batches=3,  # Visualize 3D every 3rd batch
        save_plots=True,
        output_dir="./visualizations_3d_dedicated",
        show_labels=False,
        max_people=2
    )
    
    # Create pipeline with dedicated visualizers
    pipeline = EstimationPipe(
        preprocessor=preprocessor,
        flatpose=flatpose,
        poselifting=poselifting,
        output_saver=output_saver,
        visualizer_2d=visualizer_2d,  # Dedicated 2D visualizer for flatpose
        visualizer_3d=visualizer_3d   # Dedicated 3D visualizer for poselifting
    )
    
    return pipeline

def create_pipeline_2d_only():
    """Example of pipeline with only 2D visualization"""
    
    # Create your actual estimation modules
    # preprocessor = YourPreprocessor()
    # flatpose = YourFlatpose() 
    # poselifting = YourPoselifting()
    # output_saver = YourOutputSaver()
    
    # Create only dedicated 2D visualizer
    visualizer_2d = Pose2DVisualizer(
        visualize_every_n_batches=1,  # Visualize every batch
        save_plots=True,
        output_dir="./visualizations_2d_only",
        show_labels=True,
        max_people=2
    )
    
    # Create pipeline with only 2D visualization
    pipeline = EstimationPipe(
        preprocessor=preprocessor,
        flatpose=flatpose,
        poselifting=poselifting,
        output_saver=output_saver,
        visualizer_2d=visualizer_2d,  # Only flatpose will be visualized
        visualizer_3d=None            # No 3D visualization
    )
    
    return pipeline


def create_pipeline_3d_only():
    """Example of pipeline with only 3D visualization"""
    
    # Create your actual estimation modules
    # preprocessor = YourPreprocessor()
    # flatpose = YourFlatpose() 
    # poselifting = YourPoselifting()
    # output_saver = YourOutputSaver()
    
    # Create only dedicated 3D visualizer
    visualizer_3d = Pose3DVisualizer(
        visualize_every_n_batches=1,  # Visualize every batch
        save_plots=True,
        output_dir="./visualizations_3d_only",
        show_labels=True,
        max_people=4
    )
    
    # Create pipeline with only 3D visualization
    pipeline = EstimationPipe(
        preprocessor=preprocessor,
        flatpose=flatpose,
        poselifting=poselifting,
        output_saver=output_saver,
        visualizer_2d=None,           # No 2D visualization
        visualizer_3d=visualizer_3d   # Only poselifting will be visualized
    )
    
    return pipeline


def run_pipeline_examples():
    """Example of running different pipeline configurations"""
    
    print("=== Recommended Pipeline Configurations ===\n")
    
    pipeline_dedicated = create_pipeline_with_dedicated_visualizers()
    print("   - Flatpose stage: Uses Pose2DVisualizer (advanced 2D features)")
    print("   - Poselifting stage: Uses Pose3DVisualizer (professional 3D rendering)")
    print("   - Each visualizer is specialized for its task")
    print("   - Best performance and features\n")
    
    print("3. Specialized: Only 2D visualization")
    pipeline_2d = create_pipeline_2d_only()
    print("   - Flatpose stage: Uses Pose2DVisualizer")
    print("   - Poselifting stage: No visualization")
    print("   - Perfect for 2D pose analysis\n")
    
    print("4. Specialized: Only 3D visualization")
    pipeline_3d = create_pipeline_3d_only()
    print("   - Flatpose stage: No visualization")
    print("   - Poselifting stage: Uses Pose3DVisualizer")
    print("   - Perfect for 3D pose analysis\n")


def compare_visualizers():
    """Compare the different visualizer options"""
    
    print("=== Visualizer Comparison ===\n")
    
    print("📊 Pose2DVisualizer (Dedicated 2D):")
    print("   ✓ Body-part color coding (head=red, arms=green/orange, etc.)")
    print("   ✓ Image overlay capabilities")
    print("   ✓ Advanced 2D styling with borders and labels")
    print("   ✓ Supports up to 4 people")
    print("   ✓ NaN handling for invalid keypoints")
    print("   ✓ OpenCV integration for overlays")
    print("   ❌ No 3D visualization\n")
    
    print("🎯 Pose3DVisualizer (Dedicated 3D):")
    print("   ✓ Professional 3D rendering")
    print("   ✓ Body-part color coding")
    print("   ✓ Advanced 3D styling with proper panes")
    print("   ✓ Optimal viewing angles")
    print("   ✓ NaN handling for invalid keypoints")
    print("   ✓ Supports up to 4 people")
    print("   ❌ No 2D visualization\n")
    
    print("🔧 SimplePoseVisualizer (Legacy):")
    print("   ✓ Handles both 2D and 3D")
    print("   ✓ Backward compatibility")
    print("   ✓ Simple and straightforward")
    print("   ❌ Less sophisticated than dedicated visualizers")
    print("   ❌ Generic styling")
    print("   ❌ Limited to 2 people\n")
    
    print("💡 Recommendation:")
    print("   Use dedicated visualizers (Pose2DVisualizer + Pose3DVisualizer)")
    print("   for new projects. They provide better features and performance.")


# Expected tensor shapes at each stage:
# 
# Input batch: Your input format (could be images, videos, etc.)
# After preprocessor: Depends on your preprocessing (could be features, normalized data, etc.)
# After flatpose: (batch_size, frames, 17, 2) - 2D keypoints for 17 joints
# After poselifting: (batch_size, frames, 17, 3) - 3D keypoints for 17 joints (final output)


if __name__ == "__main__":
    print("🎨 Pose Estimation Pipeline with Clean Visualization Architecture")
    print("=" * 65)
    
    print("\n🚀 Key Improvements:")
    print("✓ Dedicated visualizers for 2D and 3D (no more redundancy)")
    print("✓ SimplePoseVisualizer now clearly marked as legacy")
    print("✓ Clean separation of concerns")
    print("✓ Each visualizer specialized for its task")
    print("✓ Better performance (only load what you need)")
    print("✓ More advanced features in dedicated visualizers")
    
    run_pipeline_examples()
    
    print("\n" + "=" * 65)
    compare_visualizers()