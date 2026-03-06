"""
Root Cause Classifier for Airbag Recalls

Parses recall defect descriptions and classifies the underlying technical
root cause at the component level. Uses keyword-based pattern matching to
categorize failures into standardized categories.

Assumptions:
    - Defect descriptions in the NHTSA data are in English.
    - Keywords are case-insensitive.
    - A single recall may map to multiple root causes; the primary
      (first matched) cause is used for aggregation.
"""

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Root cause classification rules: (category, subsystem, component, keywords)
# Each rule is checked against the defect_description field.
CLASSIFICATION_RULES = [
    {
        "cause_category": "Inflator Rupture",
        "subsystem": "Inflator Assembly",
        "failing_component": "Inflator",
        "failure_mechanism": "Excessive internal pressure causing metal housing fracture",
        "keywords": [
            r"inflator.*ruptur",
            r"ruptur.*inflator",
            r"inflator.*explod",
            r"inflator.*fragment",
            r"inflator.*shrapnel",
            r"metal fragment",
        ],
    },
    {
        "cause_category": "Propellant Degradation",
        "subsystem": "Inflator Assembly",
        "failing_component": "Propellant Charge",
        "failure_mechanism": "Chemical degradation of ammonium nitrate propellant",
        "keywords": [
            r"propellant.*degrad",
            r"ammonium nitrate",
            r"propellant.*moisture",
            r"propellant.*deteriorat",
            r"chemical.*break\s*down",
            r"phase.*stabil",
        ],
    },
    {
        "cause_category": "Sensor Malfunction",
        "subsystem": "Crash Sensing System",
        "failing_component": "Crash Sensor",
        "failure_mechanism": "Sensor fails to detect or incorrectly detects crash event",
        "keywords": [
            r"sensor.*malfunc",
            r"sensor.*fail",
            r"crash.*sensor",
            r"impact.*sensor",
            r"sensor.*defect",
            r"accelerometer",
            r"sensing.*system",
        ],
    },
    {
        "cause_category": "Electrical Wiring Failure",
        "subsystem": "Electrical System",
        "failing_component": "Wiring Harness",
        "failure_mechanism": "Short circuit, open circuit, or wiring degradation",
        "keywords": [
            r"wir(?:e|ing).*fail",
            r"short.*circuit",
            r"open.*circuit",
            r"electrical.*connect",
            r"wir(?:e|ing).*harness",
            r"wir(?:e|ing).*chaf",
            r"electrical.*fault",
        ],
    },
    {
        "cause_category": "Control Module Software Error",
        "subsystem": "Electronic Control Unit",
        "failing_component": "ACM/RCM Software",
        "failure_mechanism": "Software bug or calibration error in airbag control module",
        "keywords": [
            r"software.*error",
            r"software.*defect",
            r"control.*module.*software",
            r"software.*update",
            r"calibration.*error",
            r"firmware",
            r"programming.*error",
            r"ACM",
            r"RCM.*module",
            r"control\s*module",
        ],
    },
    {
        "cause_category": "Connector Fault",
        "subsystem": "Electrical System",
        "failing_component": "Electrical Connector",
        "failure_mechanism": "Loose, corroded, or improperly seated connector",
        "keywords": [
            r"connector.*fault",
            r"connector.*loose",
            r"connector.*corrod",
            r"clock\s*spring",
            r"connector.*disconnect",
            r"connector.*contact",
            r"plug.*connect",
        ],
    },
    {
        "cause_category": "Manufacturing Defect",
        "subsystem": "Various",
        "failing_component": "Various",
        "failure_mechanism": "Assembly or production process error during manufacturing",
        "keywords": [
            r"manufactur.*defect",
            r"assembly.*error",
            r"production.*defect",
            r"weld.*defect",
            r"crimp.*defect",
            r"improperly.*manufactur",
            r"improperly.*assembl",
            r"incorrect.*assembl",
            r"faulty.*manufactur",
            r"misassembl",
        ],
    },
    {
        "cause_category": "Material Fatigue",
        "subsystem": "Various",
        "failing_component": "Structural Component",
        "failure_mechanism": "Material weakening due to cyclic stress or aging",
        "keywords": [
            r"material.*fatigue",
            r"metal.*fatigue",
            r"stress.*crack",
            r"corrosion",
            r"rust",
            r"wear.*tear",
            r"age.*degrad",
            r"deteriorat",
        ],
    },
    {
        "cause_category": "Inflator Over-Pressurization",
        "subsystem": "Inflator Assembly",
        "failing_component": "Inflator",
        "failure_mechanism": "Excessive gas generation leading to over-aggressive deployment",
        "keywords": [
            r"over.*pressur",
            r"excessive.*force",
            r"aggressive.*deploy",
            r"too.*much.*force",
            r"excessive.*gas",
        ],
    },
    {
        "cause_category": "Airbag Non-Deployment",
        "subsystem": "Deployment System",
        "failing_component": "Deployment Mechanism",
        "failure_mechanism": "Airbag fails to deploy during a crash event",
        "keywords": [
            r"fail.*deploy",
            r"not.*deploy",
            r"non.*deploy",
            r"may not.*inflate",
            r"does not.*inflate",
            r"will not.*deploy",
        ],
    },
    {
        "cause_category": "Inadvertent Deployment",
        "subsystem": "Deployment System",
        "failing_component": "Deployment Trigger",
        "failure_mechanism": "Airbag deploys without a crash event",
        "keywords": [
            r"inadvert.*deploy",
            r"unexpect.*deploy",
            r"without.*crash",
            r"unintend.*deploy",
            r"spontaneous.*deploy",
            r"inadvert.*inflat",
            r"unexpect.*inflat",
        ],
    },
    {
        "cause_category": "Cushion/Bag Defect",
        "subsystem": "Airbag Cushion",
        "failing_component": "Airbag Cushion",
        "failure_mechanism": "Tear, weakness, or improper folding of the airbag fabric",
        "keywords": [
            r"cushion.*tear",
            r"bag.*tear",
            r"fabric.*defect",
            r"seam.*fail",
            r"cushion.*defect",
            r"fold.*incorrect",
            r"bag.*fold",
            r"tether",
        ],
    },
    {
        "cause_category": "Supplier Component Defect",
        "subsystem": "Various",
        "failing_component": "Supplier Part",
        "failure_mechanism": "Defective component from a tier supplier",
        "keywords": [
            r"supplier.*defect",
            r"supplier.*component",
            r"takata",
            r"autoliv",
            r"delphi",
            r"continental",
            r"joyson",
            r"ARC.*automotive",
        ],
    },
    {
        "cause_category": "Label/Documentation Error",
        "subsystem": "Documentation",
        "failing_component": "Warning Label",
        "failure_mechanism": "Incorrect or missing airbag labels or documentation",
        "keywords": [
            r"label.*incorrect",
            r"label.*missing",
            r"document.*error",
            r"owner.*manual",
            r"label.*error",
            r"wrong.*label",
        ],
    },
    {
        "cause_category": "Occupant Classification Error",
        "subsystem": "Occupant Detection System",
        "failing_component": "OCS Sensor/Module",
        "failure_mechanism": "Incorrect occupant weight/position classification",
        "keywords": [
            r"occupant.*classif",
            r"occupant.*detect",
            r"weight.*sensor",
            r"passenger.*detect",
            r"occupant.*sensor",
            r"child.*detect",
            r"seat.*sensor",
        ],
    },
]

# Supplier identification patterns
SUPPLIER_PATTERNS = {
    "Takata": [r"takata", r"TK\s*Holdings"],
    "Autoliv": [r"autoliv"],
    "Delphi": [r"delphi"],
    "Continental": [r"continental"],
    "Joyson Safety Systems": [r"joyson", r"key\s*safety"],
    "ARC Automotive": [r"ARC\s*automotiv"],
    "ZF/TRW": [r"\bZF\b", r"\bTRW\b"],
}


def classify_root_cause(description: str) -> dict:
    """Classify a single recall description into root cause categories.

    Args:
        description: The defect description text.

    Returns:
        Dict with cause_category, subsystem, failing_component,
        failure_mechanism, and supplier fields.
    """
    if not isinstance(description, str) or not description.strip():
        return {
            "cause_category": "Unclassified",
            "subsystem": "Unknown",
            "failing_component": "Unknown",
            "failure_mechanism": "Insufficient description data",
            "supplier": "Unknown",
        }

    text = description.lower()
    matched_cause = None

    for rule in CLASSIFICATION_RULES:
        for pattern in rule["keywords"]:
            if re.search(pattern, text, re.IGNORECASE):
                matched_cause = rule
                break
        if matched_cause:
            break

    # Identify supplier
    supplier = "Unknown"
    for supplier_name, patterns in SUPPLIER_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                supplier = supplier_name
                break
        if supplier != "Unknown":
            break

    if matched_cause:
        return {
            "cause_category": matched_cause["cause_category"],
            "subsystem": matched_cause["subsystem"],
            "failing_component": matched_cause["failing_component"],
            "failure_mechanism": matched_cause["failure_mechanism"],
            "supplier": supplier,
        }

    return {
        "cause_category": "Unclassified",
        "subsystem": "Unknown",
        "failing_component": "Unknown",
        "failure_mechanism": "Could not determine from description",
        "supplier": supplier,
    }


def classify_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply root cause classification to an entire DataFrame.

    Args:
        df: DataFrame with a 'defect_description' column.

    Returns:
        DataFrame with additional classification columns added.
    """
    logger.info("Classifying root causes for %d records", len(df))

    classifications = df["defect_description"].apply(classify_root_cause)
    classification_df = pd.DataFrame(classifications.tolist())

    result = pd.concat([df, classification_df], axis=1)

    # Log classification summary
    counts = result["cause_category"].value_counts()
    logger.info("Classification summary:\n%s", counts.to_string())

    unclassified = (result["cause_category"] == "Unclassified").sum()
    pct = unclassified / len(result) * 100 if len(result) > 0 else 0
    logger.info("Unclassified: %d (%.1f%%)", unclassified, pct)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Quick test
    test_descriptions = [
        "The Takata inflator may rupture causing metal fragments to strike occupants",
        "Software error in the airbag control module may cause non-deployment",
        "Wiring harness may chafe causing short circuit in airbag system",
        "Ammonium nitrate propellant may degrade in high humidity conditions",
        "The airbag may inadvertently deploy without a crash event",
    ]

    for desc in test_descriptions:
        result = classify_root_cause(desc)
        print(f"Description: {desc[:60]}...")
        print(f"  Category: {result['cause_category']}")
        print(f"  Component: {result['failing_component']}")
        print(f"  Supplier: {result['supplier']}")
        print()
