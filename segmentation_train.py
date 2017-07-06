from datasets import CamVid
from models import enet, seg_net
from utils.nets import save_trained_model


evaluation_set_ratio = 0.01


camvid = CamVid.load_from_datasets_dir()
evaluation_set_size = int(evaluation_set_ratio * len(camvid))
#model = seg_net(camvid.img_size, camvid.categories_count, filter_count=16)
model_name = 'enet'
model = enet(camvid.img_size, camvid.categories_count)
model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

train_set = camvid[0:-evaluation_set_size]
model.fit_generator(train_set.generator(1),
                    len(train_set),
                    epochs=2)
save_trained_model(model, model_name)


evaluate_set = camvid[len(camvid) - evaluation_set_size:len(camvid)]
print model.evaluate_generator(evaluate_set.generator(1), evaluation_set_size)

