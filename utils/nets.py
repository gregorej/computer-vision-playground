import time
from keras.utils import plot_model


def save_trained_model(model, model_name, directory='trained_nets'):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    model.save("{}/{}.{}.h5".format(directory, model_name, timestamp))
    model.save_weights("{}/{}.{}.weights.h5".format(directory, model_name, timestamp), overwrite=True)
    plot_model(model, to_file='{}/{}.{}.png'.format(directory, model_name, timestamp), show_shapes=True)