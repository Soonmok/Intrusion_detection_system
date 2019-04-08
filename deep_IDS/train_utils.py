import tensorflow as tf

def train_step(batch_data):
    train_batch_data = sess.run(batch_data)
    feed_dict = {input_x: train_batch_data}
    _, cost, step= sess.run(
        [train_op, total_loss, global_step], feed_dict)
    return cost, step

def train_STL(epoch, train_batch):
    while True:
        try:
            cost, step = train_step(train_batch)
            if step % 100 == 0:
                print("step {}, cost {} ".format(step, cost))
        except ValueError: 
            print("==============================")
            epoch += 1
            if epoch < config.epoch:
                pass
            else:
                break

def train_classification_step(batch_data):

