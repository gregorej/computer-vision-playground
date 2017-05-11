from datasets import CamVid
from models import enet, seg_net
from keras.utils import plot_model
import time


evaluation_set_ratio = 0.01


def save_trained_model(model, model_name):
    import os
    dir = 'trained_nets'
    if not os.path.exists(dir):
        os.makedirs(dir)
    timestamp = time.strftime("%Y%m%d-%H:%M:%S")
    model.save("{}/{}.{}.h5".format(dir, model_name, timestamp))
    model.save_weights("{}/{}.{}.weights.h5".format(dir, model_name, timestamp), overwrite=True)
    plot_model(model, to_file='{}/{}.{}.png'.format(dir, model_name, timestamp), show_shapes=True)

camvid = CamVid.load_from_datasets_dir()
evaluation_set_size = int(evaluation_set_ratio * len(camvid))
#model = seg_net(camvid.img_size, camvid.categories_count, filter_count=16)
model_name = 'enet'
model = enet(camvid.img_size, camvid.categories_count)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

train_set = camvid[0:-evaluation_set_size]
model.fit_generator(train_set.generator(1),
                    nb_epoch=2,
                    samples_per_epoch=len(train_set))
save_trained_model(model, model_name)

evaluate_set = camvid[len(camvid) - evaluation_set_size:len(camvid)]
print model.evaluate_generator(evaluate_set.generator(1), evaluation_set_size)

