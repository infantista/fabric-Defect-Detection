# fabric-Defect-Detection


### Technical Process

1. **Image Acquisition**:
   - User uploads fabric image via web interface
   - System accepts JPG/PNG formats

2. **Preprocessing**:
   - Image resized to 128×128 pixels
   - Pixel values normalized (0-1 range)
   - Converted to tensor format

3. **Defect Detection**:
   - Custom CNN model processes the image
   - 7-layer architecture with dropout
   - Outputs defect probability (0-1)

4. **Result Delivery**:
   - Returns "Good" or "Defective" classification
   - Includes confidence percentage
   - Visual feedback in web UI

## Installation

### Prerequisites
- Python 3.11
- pip package manager

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fabric-defect-detection.git

### Usage
### Training the Model

python train.py

### Test the model

python predict.py

### Running the Web App

python app.py