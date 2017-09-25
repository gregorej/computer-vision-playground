from datasets import CamVid

ds = CamVid.load_from_datasets_dir()

print(ds._label_dict)

print(len(ds._label_dict))