import tensorflow as tf

def test_step(inputs, model):
    images, idnum = inputs
    
    output = model(images, training=False)
    pred = tf.argmax(output, axis=1)
    return idnum, pred

@tf.function
def distributed_test_step(strategy, dataset_inputs, model):
    return strategy.run(test_step, args=(dataset_inputs, model))