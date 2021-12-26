import tensorflow as tf

def build_model(class_num, image_size):
    pretrained_backbone = tf.keras.applications.vgg16.VGG16(
        include_top=False
        , weights='imagenet'
        , pooling='avg'
    )
    pretrained_backbone.trainable = False
    
    inputs = tf.keras.Input( shape=(*image_size, 3) ) 
    x = pretrained_backbone(inputs)
    x = tf.keras.layers.Dense(class_num, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model