from datasets import CamVid

ds = CamVid.from_dir()

print ds._label_dict

print len(ds._label_dict)