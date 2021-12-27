import tensorflow as tf

global GLOBAL_BATCH_SIZE

def train_step(inputs, model, optimizer, train_accuracy, compute_loss, GLOBAL_BATCH_SIZE):
    images, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions, GLOBAL_BATCH_SIZE)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss 

@tf.function
def distributed_train_step(strategy, dataset_inputs, model, optimizer, train_accuracy, compute_loss, GLOBAL_BATCH_SIZE):
    per_replica_losses = strategy.run(train_step, args=(dataset_inputs, model, optimizer, train_accuracy, compute_loss, GLOBAL_BATCH_SIZE))
    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)