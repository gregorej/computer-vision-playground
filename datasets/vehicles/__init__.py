from .VehiclesMaskDataset import VehiclesMaskDataset
#from predefined import custom_ds, crowdai, object_detect


def concat(datasets):
    return VehiclesMaskDataset.concat(datasets)