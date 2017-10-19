from .VehiclesMaskDataset import VehiclesMaskDataset

crowdai = VehiclesMaskDataset.load_from_dir('vehicles/object-detection-crowdai', separator=',')
object_detect = VehiclesMaskDataset.load_from_dir('vehicles/object-dataset')
custom_ds = VehiclesMaskDataset.load_from_dir('vehicles/custom')
