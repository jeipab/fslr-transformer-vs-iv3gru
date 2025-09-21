"""
Label assignment script for Filipino sign language recognition.

This script maps gloss text labels to numeric IDs using a reference file.
Converts text-based labels to integer IDs required for model training.

Usage:
    python data/splitting/assign.py

Input: data/processed/labels.csv (with text labels)
Reference: data/splitting/labels_reference.csv
Output: Updated labels.csv with numeric IDs
"""

import pandas as pd

# Load reference and labels
gloss_cat = pd.read_csv("data/splitting/labels_reference.csv")
labels = pd.read_csv("data/processed/labels.csv")

# Create mapping dictionaries
gloss_map = dict(zip(gloss_cat["label"].str.lower(), gloss_cat["gloss_id"]))
cat_map = dict(zip(gloss_cat["label"].str.lower(), gloss_cat["cat_id"]))

def get_gloss_from_filename(filename):
    """Extract gloss text from filename.
    
    Args:
        filename: Video filename
        
    Returns:
        Extracted gloss text in lowercase
    """
    name = filename.split("_", 2)[-1].replace(".npz", "").strip().lower()
    return name

# Extract gloss text from filenames
labels["gloss_text"] = labels["file"].apply(get_gloss_from_filename)

# Map text labels to numeric IDs
labels["gloss"] = labels["gloss_text"].map(gloss_map)
labels["cat"] = labels["gloss_text"].map(cat_map)

# Remove helper column
labels = labels.drop(columns=["gloss_text"])

# Save updated labels.csv
labels.to_csv("data/processed/labels.csv", index=False)

print("labels.csv has been updated with gloss_id and cat_id mappings.")
