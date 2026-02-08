from datasets import load_from_disk
ds = load_from_disk("./data/prepared/train")
print(ds.column_names)
print(ds[0].keys())