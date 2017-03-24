from CamVidDataset import CamVidDataset

ds = CamVidDataset.from_dir()

print ds._label_dict

print len(ds._label_dict)