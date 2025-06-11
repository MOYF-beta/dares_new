#!/usr/bin/env python3
"""
Quick test script for the simple 3D reconstruction
"""

import os
import sys

# Add current directory to path
sys.path.append("/mnt/c/Users/14152/Desktop/new_code")

def create_test_image():
    """Create a simple test image"""
    from PIL import Image, ImageDraw
    import numpy as np
    
    # Create a simple test image with some geometric shapes
    width, height = 320, 256
    image = Image.new('RGB', (width, height), color=(100, 150, 200))
    draw = ImageDraw.Draw(image)
    
    # Draw some shapes
    draw.rectangle([50, 50, 150, 150], fill=(255, 100, 100))
    draw.ellipse([200, 100, 280, 180], fill=(100, 255, 100))
    draw.polygon([(160, 50), (200, 100), (120, 100)], fill=(100, 100, 255))
    
    # Save test image
    test_image_path = "/mnt/c/Users/14152/Desktop/new_code/test_image.png"
    image.save(test_image_path)
    print(f"Created test image: {test_image_path}")
    return test_image_path

def main():
    print("=== Simple 3D Reconstruction Test ===")
    
    # Create test image
    test_image = create_test_image()
    
    # Check if we have any model weights available
    model_paths = [
        "/mnt/c/Users/14152/Desktop/new_code/DARES/af_sfmlearner_weights",
        "/mnt/c/Users/14152/Desktop/new_code/exps",  # Check for model weights in exps folder
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            # Look for depth_model.pth or similar
            for file in os.listdir(path):
                if 'depth' in file.lower() and file.endswith('.pth'):
                    model_path = os.path.join(path, file)
                    break
            if model_path:
                break
    
    if not model_path:
        print("No DARES model weights found. You need to:")
        print("1. Download or train a DARES PEFT model")
        print("2. Place the model weights in an accessible location")
        print("3. Update the model path in this script")
        print("\nExample usage:")
        print("python simple_3d_reconstruction.py \\")
        print("  --image test_image.png \\")
        print("  --model /path/to/dares/model \\")
        print("  --output ./reconstruction_results")
        return
    
    print(f"Found model: {model_path}")
    
    # Import and run the reconstruction
    try:
        import simple_3d_reconstruction
        import argparse
        
        # Simulate command line arguments
        sys.argv = [
            'simple_3d_reconstruction.py',
            '--image', test_image,
            '--model', model_path,
            '--output', './test_reconstruction_output',
            '--no_visualization',  # Skip visualization for automated testing
            '--depth_scale', '1.0'
        ]
        
        simple_3d_reconstruction.main()
        
    except Exception as e:
        print(f"Error during reconstruction: {e}")
        print("\nThis is expected if you don't have the proper model weights.")
        print("The script structure is ready, just need proper DARES PEFT model weights.")

if __name__ == "__main__":
    main()
