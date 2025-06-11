# Simple 3D Reconstruction with DARES PEFT

This script provides a simple interface to perform 3D reconstruction from a single image using the DARES PEFT (Parameter-Efficient Fine-Tuning) depth estimation model.

## Features

- Load DARES PEFT depth model (LoRA/DoRA)
- Predict depth from a single RGB image
- Generate 3D point cloud reconstruction
- Visualize results with Open3D
- Save reconstruction outputs (RGB, depth, point cloud)

## Dependencies

```bash
pip install torch torchvision
pip install transformers
pip install peft
pip install open3d
pip install opencv-python
pip install matplotlib
pip install pillow
pip install numpy
```

## Usage

### Basic Usage

```bash
python simple_3d_reconstruction.py \
  --image /path/to/your/image.jpg \
  --model /path/to/dares/model \
  --output ./reconstruction_output
```

### Advanced Usage

```bash
python simple_3d_reconstruction.py \
  --image input.jpg \
  --model ./trained_model \
  --output ./results \
  --min_depth 0.1 \
  --max_depth 100.0 \
  --depth_scale 5.4 \
  --fov 60.0 \
  --device cuda \
  --no_visualization
```

## Parameters

- `--image`: Path to input RGB image
- `--model`: Path to DARES PEFT model weights (directory or .pth file)
- `--output`: Output directory for results (default: ./reconstruction_output)
- `--min_depth`: Minimum depth value in meters (default: 0.1)
- `--max_depth`: Maximum depth value in meters (default: 50.0)
- `--depth_scale`: Scaling factor for depth values (default: 1.0)
- `--fov`: Camera field of view in degrees (default: 60.0)
- `--device`: Device to run on: cuda or cpu (default: cuda)
- `--no_visualization`: Skip 3D visualization window

## Model Requirements

The script expects a DARES PEFT model with the following characteristics:
- Based on DepthAnything V2 Small
- Uses LoRA or DoRA for parameter-efficient fine-tuning
- Target modules: ['query', 'value']
- Rank (r): [14,14,12,12,10,10,8,8,8,8,8,8]

## Output Files

The script generates the following output files:
- `{basename}_rgb.png`: Original RGB image
- `{basename}_depth.png`: Depth visualization (colored)
- `{basename}_depth.npy`: Raw depth values (NumPy array)
- `{basename}_pointcloud.ply`: 3D point cloud (PLY format)

## Example Model Loading

The script supports two model loading methods:

### Method 1: PEFT Directory Structure
```
model_directory/
├── adapter_config.json
├── adapter_model.safetensors
└── head.pth
```

### Method 2: Single Model File
```
depth_model.pth  # Complete model state dict
```

## Camera Intrinsics

The script automatically generates camera intrinsic parameters based on the image dimensions and specified field of view. For more accurate reconstruction, you can modify the `create_intrinsic_matrix()` function to use your actual camera parameters.

## Depth Scaling

The depth values from neural networks often need scaling to match real-world units. Common scaling factors:
- KITTI dataset: 5.4 (for stereo-trained models)
- NYU Depth: 1.0
- Custom datasets: Adjust based on your training setup

## 3D Visualization

The script uses Open3D for 3D visualization. The point cloud viewer supports:
- Mouse rotation and zoom
- Point cloud downsampling for performance
- Statistical outlier removal
- Normal estimation for better rendering

## Troubleshooting

### CUDA Out of Memory
- Use `--device cpu` to run on CPU
- Reduce image size by modifying `target_size` in `load_image()`

### Model Loading Issues
- Ensure the model path is correct
- Check that the model was trained with compatible PEFT configuration
- Verify PEFT library version compatibility

### Poor Reconstruction Quality
- Adjust `--depth_scale` parameter
- Modify `--min_depth` and `--max_depth` ranges
- Check if the input image is suitable for the trained model domain

## Integration with Existing Code

This script is based on the DARES `evaluate_3d_reconstruction.py` but simplified for single-image inference. Key differences:
- No dataset loading required
- Simplified camera intrinsics generation
- Single image processing
- Optional visualization

## Example Test

Run the test script to verify installation:

```bash
python test_3d_reconstruction.py
```

This will create a synthetic test image and attempt reconstruction (requires trained model weights).

## Related Files

- `evaluate_3d_reconstruction.py`: Original DARES 3D reconstruction evaluation
- `dares_peft.py`: DARES PEFT model implementation
- `layers.py`: Utility functions for depth conversion
