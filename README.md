# README - Tuberculosis Detection on Chest X-Rays

## Introduction
This project explores the potential of deep learning models for automated tuberculosis detection from chest X-ray images. The study focuses on analyzing and comparing existing academic research rather than implementing a fully operational system. It leverages Convolutional Neural Networks (CNN) combined with Coordinate Attention (CoordAttention) to enhance accuracy in medical image diagnosis.

## Authors
- **Mai Thanh Lam** (Student ID: 20119137)
- **Mai Hong Phong** (Student ID: 20119192)
- **Supervisor**: Dr. Huynh The Thien

## Technologies Used
- **Programming Language**: Python
- **Libraries**: TensorFlow, PyTorch, OpenCV, NumPy, Matplotlib
- **Model Architecture**: VGG16 with CoordAttention
- **Supporting Techniques**: Transfer Learning, Freezing Network, Cross-validation

## Dataset
- **Source**: Tuberculosis diagnosis dataset from Shenzhen Hospital, China
- **Format**: 662 PNG chest X-ray images, approximately 3000x3000 pixels
- **Preprocessing**: Resizing to 224x224, normalization, data augmentation

## Experimental Results
The model achieved:
- **Top-1 Accuracy**: 93.18%
- **AUC**: 97.98%
- **Recall**: 93.19%
- **Precision**: 93.18%
- **F1-score**: 93.23%

Compared to previous models like ConvNet, ResNet50, and B-CNN, the VGG16-CoordAttention model demonstrated superior performance.

## Usage
1. **Install required libraries:**
    ```bash
    pip install tensorflow torch torchvision numpy opencv-python matplotlib
    ```
2. **Run the model:**
    ```bash
    python train.py  # Train the model
    python test.py   # Test the model
    ```

## Conclusion
This project highlights the potential of using CNNs with attention mechanisms to support medical diagnosis. In the future, the research team aims to optimize the model for deployment on mobile devices.

## Contact
For any questions or suggestions, please contact via email: `maithanhlam@example.com`, `maihongphong@example.com`.
