import vehicles_dataset as ds
import models
import sys
import time


testing_gen = ds.generate_test_batch(20)

start = time.time()

model = models.small_unet(ds.img_size)
weights_path = 'model_weights.h5'
if len(sys.argv) > 1:
    weights_path = sys.argv[1]
model.load_weights(weights_path)

training_gen = ds.generate_train_batch(2)
batch_img, batch_mask = next(training_gen)

pred_all = model.predict(batch_img)
end = time.time()
print end - start
