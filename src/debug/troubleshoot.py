"""Interactive troubleshooting helper for Clinical Image Masker.

Provides an InteractiveTroubleshooter that guides users through environment
checks, synthetic tests, user-image tests, automatic fixes, and creates a
minimal working example. Designed to be robust when dependencies are missing.
"""
from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import json

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None


class InteractiveTroubleshooter:
    def __init__(self):
        self.project_root = os.getcwd()

    def run_interactive_diagnosis(self) -> None:
        """Run interactive troubleshooting session"""
        print("🏥 CLINICAL IMAGE MASKER - TROUBLESHOOTING")
        print("=" * 50)

        # Step 1: Basic environment check
        print("\n📋 Step 1: Environment Check")
        if not self.check_environment():
            ans = input("Would you like me to attempt automatic fixes? (y/n): ")
            if ans.strip().lower().startswith("y"):
                self.fix_environment()
            return

        # Step 2: Test with known working image
        print("\n🖼️  Step 2: Testing with synthetic image")
        if not self.test_synthetic_image():
            self.diagnose_synthetic_failure()
            return

        # Step 3: Test user's image
        print("\n📸 Step 3: Testing your image")
        user_image = input("Enter path to your image (or leave empty to skip): ")
        if user_image:
            if not self.test_user_image(user_image):
                self.diagnose_user_image_failure(user_image)
                return

        print("\n✅ All tests passed! Your setup is working correctly.")

    def check_environment(self) -> bool:
        """Check if environment is properly set up"""
        issues = []
        # Check Python version
        if sys.version_info < (3, 8):
            issues.append("Python version too old. Need 3.8+")

        # Check required packages
        required_packages = ["cv2", "easyocr", "numpy", "spacy"]
        for package in required_packages:
            try:
                __import__(package)
                print(f"✅ {package}: OK")
            except ImportError:
                issues.append(f"Missing package: {package}")
                print(f"❌ {package}: MISSING")

        # Check spaCy model
        try:
            import spacy as _sp
            try:
                _sp.load("en_core_web_sm")
                print("✅ spaCy model: OK")
            except Exception:
                issues.append("spaCy English model not installed")
                print("❌ spaCy model: MISSING")
        except Exception:
            # spacy import already flagged above
            pass

        if issues:
            print(f"\n❌ Found {len(issues)} issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False

        return True

    def fix_environment(self) -> None:
        """Automatically fix environment issues"""
        print("\n🔧 Attempting to fix environment issues...")
        try:
            # Use sys.executable to ensure venv pip is used
            subprocess.run([sys.executable, "-m", "pip", "install", "opencv-python", "easyocr", "spacy"], check=False)
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], check=False)
            print("✅ Environment fix attempted. Please restart the terminal/session and re-run troubleshooting.")
        except Exception as e:
            print(f"❌ Failed to fix environment: {e}")
            print("Please manually install: pip install opencv-python easyocr spacy")
            print("Then run: python -m spacy download en_core_web_sm")

    def test_synthetic_image(self) -> bool:
        """Test with a synthetic image we control"""
        print("Creating synthetic test image...")
        if np is None or cv2 is None:
            print("❌ numpy or cv2 missing; cannot create test image")
            return False

        # Create test image
        test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_image, "Patient: John Doe", (50, 100), font, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "MRN: 123456789", (50, 150), font, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "DOB: 01/15/1980", (50, 200), font, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "Phone: 555-123-4567", (50, 250), font, 1, (0, 0, 0), 2)

        # Save test image
        os.makedirs("debug_output", exist_ok=True)
        test_path = "debug_output/synthetic_test.jpg"
        cv2.imwrite(test_path, test_image)
        print(f"✅ Created test image: {test_path}")

        # Run masking using SimpleMasker when available
        try:
            from src.quick_fix.simple_masker import SimpleMasker

            masker = SimpleMasker()
            result = masker.mask_image_simple(test_path, "debug_output/synthetic_masked.jpg")
            if isinstance(result, dict) and result.get("success") and result.get("masks_applied", 0) > 0:
                print(f"✅ Synthetic test passed: {result['masks_applied']} regions masked")
                return True
            else:
                print(f"❌ Synthetic test failed: {result}")
                return False
        except Exception as e:
            print(f"❌ Synthetic test error: {e}")
            return False

    def diagnose_synthetic_failure(self) -> None:
        """Diagnose why synthetic test failed"""
        print("\n🔍 Diagnosing synthetic test failure...")
        try:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=False)
            test_image = None
            if cv2 is not None and os.path.exists("debug_output/synthetic_test.jpg"):
                test_image = cv2.imread("debug_output/synthetic_test.jpg")
            else:
                print("❌ Test image missing or cv2 not available")
                return

            results = reader.readtext(test_image)
            print(f"📊 EasyOCR found {len(results)} text regions:")
            for i, (bbox, text, confidence) in enumerate(results):
                print(f"  {i+1}. '{text}' (confidence: {confidence:.2f})")

            if len(results) == 0:
                print("❌ EasyOCR is not detecting any text!")
                print("💡 Try: Increase image contrast or check image format")
        except Exception as e:
            print(f"❌ EasyOCR test failed: {e}")

    def test_user_image(self, path: str) -> bool:
        """Run the masker on a user-provided image path"""
        if not os.path.exists(path):
            print("❌ Provided image path does not exist")
            return False
        try:
            from src.quick_fix.simple_masker import SimpleMasker
            masker = SimpleMasker()
            outpath = "debug_output/user_masked.jpg"
            result = masker.mask_image_simple(path, outpath)
            if isinstance(result, dict) and result.get("success"):
                print(f"✅ User image processed, output: {outpath}")
                return True
            print(f"❌ Masking failed: {result}")
            return False
        except Exception as e:
            print(f"❌ Error running masker: {e}")
            return False

    def diagnose_user_image_failure(self, path: str) -> None:
        """Provide guidance for user image failures"""
        print("\n🔎 Diagnosing user image failure...")
        print("Check: file exists, readable, typical image formats (jpg/png), and not corrupted.")
        print("If OCR fails: try increasing contrast, resizing to larger resolution, or using the provided minimal_working_example.py")

    def create_minimal_working_example(self) -> None:
        """Create the simplest possible working example"""
        example_code = '''
import cv2
import easyocr
import numpy as np

# Create simple test
image = cv2.imread("your_image.jpg")
reader = easyocr.Reader(['en'], gpu=False)
results = reader.readtext(image)

print(f"Found {len(results)} text regions")

# Mask all detected text with black rectangles
for (bbox, text, confidence) in results:
    bbox = np.array(bbox, dtype=np.int32)
    x, y, w, h = cv2.boundingRect(bbox)
    cv2.rectangle(image, (x-5, y-5), (x+w+5, y+h+5), (0, 0, 0), -1)

cv2.imwrite("masked_output.jpg", image)
print("Masking complete!")
'''
        with open("minimal_working_example.py", "w") as f:
            f.write(example_code)
        print("✅ Created minimal_working_example.py")
        print("Try running: python minimal_working_example.py")


def main() -> None:
    troubleshooter = InteractiveTroubleshooter()
    troubleshooter.run_interactive_diagnosis()


if __name__ == "__main__":
    main()
