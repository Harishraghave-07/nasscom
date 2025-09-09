#!/bin/bash

echo "🏥 CLINICAL IMAGE MASKER - COMPLETE RESET AND FIX"
echo "================================================"

# Step 1: Clean everything
echo "🧹 Step 1: Cleaning existing installation..."
rm -rf .venv/
rm -rf __pycache__/
rm -rf src/__pycache__/
rm -rf logs/*
rm -rf temp/*
rm -rf debug_output/*
find . -name "*.pyc" -delete

# Step 2: Create fresh environment
echo "🐍 Step 2: Creating fresh Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# Step 3: Install minimal dependencies
echo "📦 Step 3: Installing minimal working dependencies..."
pip install --upgrade pip
pip install opencv-python==4.8.1.78
pip install easyocr==1.7.0
pip install numpy==1.24.3
pip install spacy==3.6.1

# Step 4: Download models
echo "🤖 Step 4: Downloading required models..."
python -m spacy download en_core_web_sm

# Step 5: Create directory structure
echo "📁 Step 5: Creating directory structure..."
mkdir -p data/{raw,processed}
mkdir -p logs
mkdir -p temp
mkdir -p debug_output/{stage1,stage2,stage3,stage4}
mkdir -p src/{core,preprocessing,ocr,phi_detection,masking,debug,quick_fix}

# Step 6: Create minimal working implementation
echo "⚡ Step 6: Creating minimal working implementation..."
cat > quick_test.py << 'EOF'
import cv2
import easyocr
import numpy as np
import os

def quick_mask_test(image_path):
    print(f"Testing with image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Could not load image")
        return False
    
    print(f"✅ Image loaded: {image.shape}")
    
    # Initialize EasyOCR
    reader = easyocr.Reader(['en'], gpu=False)
    print("✅ EasyOCR initialized")
    
    # Detect text
    results = reader.readtext(image)
    print(f"✅ Found {len(results)} text regions")
    
    if len(results) == 0:
        print("❌ No text detected")
        return False
    
    # Mask all text with black rectangles
    masked = image.copy()
    for (bbox, text, confidence) in results:
        print(f"  Masking: '{text}' (confidence: {confidence:.2f})")
        bbox = np.array(bbox, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(bbox)
        cv2.rectangle(masked, (x-3, y-3), (x+w+3, y+h+3), (0, 0, 0), -1)
    
    # Save result
    output_path = "debug_output/quick_test_result.jpg"
    cv2.imwrite(output_path, masked)
    print(f"✅ Result saved to: {output_path}")
    
    # Save side-by-side comparison
    comparison = np.hstack((image, masked))
    cv2.imwrite("debug_output/comparison.jpg", comparison)
    print("✅ Comparison saved to: debug_output/comparison.jpg")
    
    return True

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        success = quick_mask_test(sys.argv[1])
    else:
        # Create test image
        test_img = np.ones((300, 500, 3), dtype=np.uint8) * 255
        cv2.putText(test_img, "John Doe", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(test_img, "MRN-123456", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imwrite("debug_output/test_input.jpg", test_img)
        success = quick_mask_test("debug_output/test_input.jpg")
    
    if success:
        print("🎉 SUCCESS! Basic masking is working!")
    else:
        print("❌ FAILED! Check the error messages above.")
EOF

# Step 7: Run test
echo "🧪 Step 7: Running quick test..."
python quick_test.py

echo ""
echo "🎯 RESET COMPLETE!"
echo "Next steps:"
echo "1. Run: python quick_test.py your_image.jpg"
echo "2. Check debug_output/ for results"
echo "3. If working, build upon this minimal foundation"
