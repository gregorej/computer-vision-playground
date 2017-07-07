from vehicle_recognition import vehicles_dataset as ds
from models import small_unet
from keras import backend as K
from keras.optimizers import Adam
from utils.nets import save_trained_model


def IOU_calc(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return 2*(intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def IOU_calc_loss(y_true, y_pred):
    return -IOU_calc(y_true, y_pred)

training_gen = ds.generate_train_batch(1)
smooth = 1.
model = small_unet((ds.img_rows, ds.img_cols))
model.compile(optimizer=Adam(lr=1e-4),
              loss=IOU_calc_loss, metrics=[IOU_calc])

history = model.fit_generator(training_gen, steps_per_epoch=5000,  epochs=1)

save_trained_model(model, 'small_unet')
