# 光流和外观流可视化脚本使用说明

这个项目提供了两个脚本来可视化position（光流）和transformation（外观流）：

## 脚本文件

1. **`visualize_flows_simple.py`** - 简化版本，推荐使用
2. **`visualize_position_transform.py`** - 完整版本，功能更全面

## 安装依赖

```bash
pip install torch torchvision opencv-python matplotlib Pillow numpy
```

## 使用方法

### 基本用法

```bash
python visualize_flows_simple.py \
    --img1 path/to/image1.jpg \
    --img2 path/to/image2.jpg \
    --model_path path/to/trained/models
```

### 参数说明

- `--img1`: 参考图像路径（第一张图）
- `--img2`: 目标图像路径（第二张图）  
- `--model_path`: 训练好的模型权重文件夹路径
- `--height`: 图像高度，默认256
- `--width`: 图像宽度，默认320
- `--save_dir`: 结果保存目录，默认`./flow_visualization`
- `--device`: 计算设备，默认`cuda`

### 示例

```bash
# 使用默认参数
python visualize_flows_simple.py \
    --img1 ./data/img1.jpg \
    --img2 ./data/img2.jpg \
    --model_path ./logs/dsfm_experiment/models/weights_100

# 指定图像尺寸和保存路径
python visualize_flows_simple.py \
    --img1 ./data/img1.jpg \
    --img2 ./data/img2.jpg \
    --model_path ./models/pretrained \
    --height 480 \
    --width 640 \
    --save_dir ./results/flow_vis \
    --device cuda
```

## 模型文件要求

模型路径文件夹中需要包含以下权重文件：

```
model_path/
├── position_encoder.pth    # 光流编码器
├── position.pth           # 光流解码器
├── transform_encoder.pth  # 外观流编码器
└── transform.pth         # 外观流解码器
```

## 输出结果

脚本会生成以下文件：

### 可视化图像
- `flow_visualization_complete.png` - 完整的可视化结果图
- `img1_reference.png` - 参考图像
- `img2_target.png` - 目标图像
- `optical_flow_color.png` - 光流颜色可视化
- `registered_image.png` - 配准后的图像
- `appearance_flow.png` - 外观流可视化
- `refined_result.png` - 最终精细化结果

### 可视化内容说明

1. **光流 (Optical Flow/Position)**
   - 2通道的位移场，表示像素在两帧间的运动
   - 颜色编码：色调表示方向，亮度表示幅度

2. **外观流 (Appearance Flow/Transform)**  
   - 3通道的外观变换，用于修正配准后的残差
   - 范围通常在[-1, 1]之间

3. **图像配准 (Image Registration)**
   - 使用光流将目标图像配准到参考图像空间

4. **精细化结果 (Refined Result)**
   - 配准图像 + 外观流的最终结果

## 工作原理

脚本模拟了DSFM训练器的推理过程：

1. **光流计算**: 使用`position_encoder`和`position`模型计算两图像间的光流
2. **图像配准**: 使用`SpatialTransformer`根据光流配准图像
3. **外观流计算**: 使用`transform_encoder`和`transform`模型计算外观变换
4. **结果精细化**: 将外观流应用到配准图像上得到最终结果

## 注意事项

1. **GPU内存**: 确保有足够的GPU内存，大图像可能需要调整batch_size
2. **图像格式**: 支持常见图像格式（jpg, png等）
3. **模型兼容性**: 确保模型权重与当前代码版本兼容
4. **依赖版本**: 如果遇到问题，检查PyTorch版本兼容性

## 故障排除

### 常见错误

1. **模型文件找不到**
   ```
   ⚠ 未找到 position_encoder.pth
   ```
   检查模型路径是否正确，确认所需的.pth文件存在

2. **CUDA内存不足**
   ```
   RuntimeError: CUDA out of memory
   ```
   尝试使用CPU：`--device cpu` 或减小图像尺寸

3. **图像尺寸不匹配**
   ```
   RuntimeError: size mismatch
   ```
   检查模型训练时使用的图像尺寸，调整`--height`和`--width`参数

### 性能优化

1. 使用较小的图像尺寸进行快速测试
2. 确保使用GPU加速（如果可用）
3. 关闭不必要的可视化以节省内存

## 扩展功能

可以修改脚本来：
- 支持视频序列的批量处理
- 添加更多的可视化选项
- 集成到其他工作流中
- 添加定量评估指标

## 联系支持

如果遇到问题，请检查：
1. 依赖版本是否正确
2. 模型文件是否完整
3. 输入图像是否有效
4. 设备（CPU/GPU）是否可用
