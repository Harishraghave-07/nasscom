
# Minimal working configuration for Clinical Image Masker

class Config:
    # OCR Settings - Very permissive for initial testing
    OCR_CONFIDENCE_THRESHOLD = 0.01  # Accept almost everything
    OCR_MIN_TEXT_AREA = 25  # Very small minimum
    OCR_LANGUAGES = ['en']
    OCR_GPU_ENABLED = False  # Force CPU

    # PHI Detection - Basic patterns only
    ENABLE_REGEX_PATTERNS = True
    ENABLE_SPACY_NER = True
    SPACY_MODEL = 'en_core_web_sm'

    # Basic PHI patterns
    PHI_PATTERNS = {
        'name': r'[A-Z][a-z]+ [A-Z][a-z]+',
        'mrn': r'MRN[-\s]?\d{6,}',
        'ssn': r'\d{3}[-\s]?\d{2}[-\s]?\d{4}',
        'phone': r'\d{3}[-\s]?\d{3}[-\s]?\d{4}'
    }

    # Masking Settings - Simple and reliable
    MASKING_METHOD = 'rectangle'
    INPAINTING_METHOD = 'telea'
    MASK_EXPANSION_PIXELS = 3

    # Logging
    LOG_LEVEL = 'DEBUG'
    ENABLE_AUDIT_LOGGING = True

    # File paths
    TEMP_DIR = 'temp'
    LOG_DIR = 'logs'
    DEBUG_OUTPUT_DIR = 'debug_output'
