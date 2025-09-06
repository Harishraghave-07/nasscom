# PHI Patterns Reference

This document lists the 18 HIPAA Safe Harbor identifiers, recommended detection method, and an example for each. Use these patterns in `PHI detection` for regex, NER, and heuristic tuning.

| # | Identifier | Detection method | Example |
|---|------------|------------------|---------|
| 1 | Names (full name) | NER (SpaCy/custom model) + name dictionaries | "John A. Smith" |
| 2 | Geographic subdivisions smaller than a state (city, county, ZIP) | Regex for ZIP codes; NER for city/place names; geodb lookup | "Springfield, IL" / "02139" |
| 3 | All elements of dates (except year) related to an individual | Regex for date formats; contextual NER | "02/14/1990" or "Feb 14" |
| 4 | Telephone numbers | Regex + libphonenumber validation | "+1 (555) 123-4567" |
| 5 | Fax numbers | Regex + libphonenumber | "+1 555-765-4321" |
| 6 | Electronic mail addresses | Regex (RFC-like) + entropy checks | "user@example.com" |
| 7 | Social security numbers (SSN) | Strict regex with checksum heuristics | "123-45-6789" |
| 8 | Medical record numbers | Regex patterns (project-specific) + whitelist/blacklist | "MRN: 987654321" |
| 9 | Health plan beneficiary numbers | Regex / structured ID patterns | "HPB-00012345" |
| 10 | Account numbers | Regex, length heuristics, checksum (if any) | "Acct: 4000123412341234" |
| 11 | Certificate/license numbers | Regex & dictionary for known issuers | "LIC-AB123456" |
| 12 | Vehicle identifiers and serial numbers, including license plates | Regex for plate formats by region; OCR confidence threshold | "KA-01-AB-1234" |
| 13 | Device identifiers and serial numbers | Regex and vendor patterns | "SN: 5XJ3E1EA7LF000000" |
| 14 | URLs | Regex (http/https) and domain checks | "https://hospital.example.org/patient/123" |
| 15 | IP addresses | Regex (IPv4/IPv6) | "192.168.1.100" |
| 16 | Biometric identifiers (fingerprints, voiceprints) | ML classifier for image patterns or metadata flags | "Fingerprint scan image" |
| 17 | Full-face photos and comparable images | Face detection on image regions; flag for manual review | "photo.jpg (face detected)" |
| 18 | Any other unique identifying number, characteristic, or code | Heuristics, entropy checks, manual whitelist/blacklist | "UniqueID: Xk3-P9q" |

Notes

- For image-based PHI, combine OCR text with bounding boxes and visual detectors (faces, logos).
- Maintain a configurable whitelist/blacklist to reduce false positives (e.g., hospital names allowed).
- Keep a `detect_phi` test suite with sample images and expected detections to validate regex/NER changes.
