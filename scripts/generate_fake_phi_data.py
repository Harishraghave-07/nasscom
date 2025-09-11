"""Generate synthetic images containing PHI-like text for pipeline testing.

Creates PNG images in `synthetic_data/` with simple layouts and PHI examples
(email, phone, SSN, name). Useful for CI e2e validation steps.
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

OUT = Path("synthetic_data")
OUT.mkdir(exist_ok=True)

samples = [
    {
        "name": "John Doe",
        "email": "john.doe@example.com",
        "phone": "+1-555-123-4567",
        "ssn": "123-45-6789",
    },
    {
        "name": "Jane Smith",
        "email": "jane.smith@health.org",
        "phone": "+1 (555) 987-6543",
        "ssn": "987-65-4321",
    },
    {
        "name": "Emily Dawson",
        "email": "emily.dawson@clinic.net",
        "phone": "+44 20 7946 0958",
        "ssn": "111-22-3333",
    },
]

# try to pick a reasonable font
try:
    FONT = ImageFont.truetype("DejaVuSans.ttf", 24)
except Exception:
    FONT = ImageFont.load_default()

for i, s in enumerate(samples, start=1):
    img = Image.new("RGB", (1200, 800), color=(255, 255, 255))
    d = ImageDraw.Draw(img)
    margin = 50
    y = margin
    lines = [
        f"Patient Name: {s['name']}",
        f"Email: {s['email']}",
        f"Phone: {s['phone']}",
        f"SSN: {s['ssn']}",
        "Diagnosis: Hypertension\nNotes: Follow-up in 2 weeks.",
    ]
    for line in lines:
        for sub in line.split('\n'):
            d.text((margin, y), sub, font=FONT, fill=(0, 0, 0))
            # measure height robustly across Pillow versions
            try:
                bbox = d.textbbox((margin, y), sub, font=FONT)
                h = bbox[3] - bbox[1]
            except Exception:
                try:
                    h = d.textsize(sub, font=FONT)[1]
                except Exception:
                    try:
                        h = FONT.getsize(sub)[1]
                    except Exception:
                        h = 20
            y += h + 12
        y += 6

    # draw a fake header and footer
    d.text((margin, 10), "ACME Clinic - Medical Report", font=FONT, fill=(0, 0, 128))
    d.text((margin, img.height - 40), "Confidential", font=FONT, fill=(128, 0, 0))

    out_path = OUT / f"fake_phi_{i:03d}.png"
    img.save(out_path)
    print("Wrote", out_path)

print("Generated", len(samples), "synthetic images in", OUT)
