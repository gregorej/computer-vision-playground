import time
from keras.utils import plot_model


def save_list_to_file(file_name, l):
    with open(file_name, 'w') as f:
        for item in l:
            f.write("{}\n".format(item))


def ensure_dir_exists(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_trained_model(model, model_name, directory='trained_nets', history=None):
    ensure_dir_exists(directory)
    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    model.save("{}/{}.{}.h5".format(directory, model_name, timestamp))
    model.save_weights("{}/{}.{}.weights.h5".format(directory, model_name, timestamp), overwrite=True)
    plot_model(model, to_file='{}/{}.{}.png'.format(directory, model_name, timestamp), show_shapes=True)
    if history is not None:
        loss_file_name = "{}/{}.{}.loss.txt".format(directory, model_name, timestamp)
        save_list_to_file(loss_file_name, history['loss'])

