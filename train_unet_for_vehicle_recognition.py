from models import small_unet
from keras import backend as K
from keras.optimizers import Adam
from utils.nets import save_trained_model
from datasets.vehicles.VehiclesMaskDataset import VehiclesMaskDataset
from datasets.vehicles.augmentations import stretch, flip_horizontally


def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)

dataset_names = ['sosnowiecka', 'smetna']
ds = VehiclesMaskDataset.concat(map(lambda name: VehiclesMaskDataset.load_from_dir(name), dataset_names))
ds = ds.shuffled()
ds = ds.with_augmentations([flip_horizontally, stretch(80)])
batch_size = 8
train_dataset = ds[0:len(ds) * 80 / 100]
valid_dataset = ds[len(ds) * 80 / 100:len(ds) * 90 / 100]
test_dataset = ds[len(ds) * 90 / 100:-1]
training_gen = train_dataset.generator(batch_size=batch_size)
smooth = 1.
model = small_unet((ds.img_rows, ds.img_cols))
model.compile(optimizer=Adam(lr=1e-4),
              loss=IOU_calc_loss, metrics=[IOU_calc])

#training_steps = len(train_dataset) / batch_size
training_steps = 1
history = model.fit_generator(training_gen,
                              steps_per_epoch=training_steps,
                              epochs=2,
                              validation_data=valid_dataset.generator(batch_size=batch_size),
                              validation_steps=len(valid_dataset) / batch_size)
save_trained_model(model, 'small_unet', history=history.history)


print history.history
print history.history['val_loss']

print "Evaluating..."

print model.evaluate_generator(test_dataset.generator(batch_size=batch_size),
                               steps=len(test_dataset)/2)

