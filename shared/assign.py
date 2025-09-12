import pandas as pd

# Load reference and labels
gloss_cat = pd.read_csv("shared/labels_reference.csv")   # the file with gloss_id, label, cat_id, category
labels = pd.read_csv("shared/sample_processed/labels.csv")   # CHANGE THIS to the actual location of the labels.csv file

# Create mapping dictionaries
gloss_map = dict(zip(gloss_cat["label"].str.lower(), gloss_cat["gloss_id"]))
cat_map = dict(zip(gloss_cat["label"].str.lower(), gloss_cat["cat_id"]))

# Extract gloss text from filename
def get_gloss_from_filename(filename):
    name = filename.split("_", 2)[-1].replace(".npz", "").strip().lower()
    return name

labels["gloss_text"] = labels["file"].apply(get_gloss_from_filename)

# Replace gloss and cat with correct IDs
labels["gloss"] = labels["gloss_text"].map(gloss_map)
labels["cat"]   = labels["gloss_text"].map(cat_map)

# Drop helper column
labels = labels.drop(columns=["gloss_text"])

# Save updated labels.csv in shared folder
labels.to_csv("shared/labels_updated.csv", index=False)

print(" labels_updated.csv has been created in the shared folder using the new reference table.")
