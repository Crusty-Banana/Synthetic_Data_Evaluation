from datasets import load_dataset

ds = load_dataset("uitnlp/vietnamese_students_feedback")

for split in ds.keys():
    ds[split].to_csv(f"vietnamese_students_feedback_{split}.csv", index=False)
    print(f"Saved {split} split to vietnamese_students_feedback_{split}.csv")
