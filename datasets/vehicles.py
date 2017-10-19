from .vehicles.VehiclesMaskDataset import VehiclesMaskDataset


def concat(datasets):
    return VehiclesMaskDataset.concat(datasets)

crowdai = VehiclesMaskDataset.load_from_dir('object-detection-crowdai', separator=',')
object_detect = VehiclesMaskDataset.load_from_dir('object-dataset')