# PHI Reference

This reference expands the PHI patterns used by the project. Use these as starting points for regex and model tuning.

1. Names (Full name)
   - Method: NER (SpaCy) + name dictionaries
   - Example: John A. Smith
   - Regex hint: (?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)

2. Geographic subdivisions (city, county, ZIP)
   - Method: Regex for ZIP, NER for place names
   - Example: Springfield, IL / 02139
   - Regex hint: \b\d{5}(?:-\d{4})?\b

3. Dates
   - Method: Regex + contextual parsing
   - Example: 02/14/1990, Feb 14
   - Regex hint: \b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b

4. Telephone
   - Method: libphonenumber validation
   - Example: +1 (555) 123-4567
   - Regex hint: (?:\+?\d{1,3}[\s-]?)?(?:\(\d{3}\)|\d{3})[\s-]?\d{3}[\s-]?\d{4}

5. Fax
   - Method: Same as telephone

6. Email
   - Method: RFC-compatible regex + entropy check
   - Example: user@example.com

7. SSN
   - Method: Strict regex with heuristics
   - Example: 123-45-6789
   - Regex hint: \b\d{3}-\d{2}-\d{4}\b

8. Medical Record Numbers
   - Method: Project-specific patterns + whitelist
   - Example: MRN: 987654321

9. Health Plan Beneficiary Numbers
   - Method: Regex
   - Example: HPB-00012345

10. Account Numbers
    - Method: Length heuristics, Luhn where applicable

11. Certificate/License Numbers
    - Method: Regex + issuer dictionary

12. Vehicle identifiers / license plates
    - Method: Regex depending on region

13. Device serial numbers
    - Method: Vendor patterns

14. URLs
    - Method: Regex, domain allowlist/denylist

15. IP addresses
    - Method: IPv4/v6 regex

16. Biometric identifiers
    - Method: ML-based detection + metadata flags

17. Full-face photos
    - Method: Face detection; flag for manual review

18. Other unique identifiers
    - Method: Entropy checks, heuristics


Notes:
- Maintain a centralized `PHI_PATTERNS` config that maps identifiers to detection functions.
- Keep sample datasets and unit tests for each pattern to reduce regressions.
