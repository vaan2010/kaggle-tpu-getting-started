import tensorflow as tf
from tensorflow.keras import layers, Model, Sequential

# resnet-18，34的结构
class BasicBlock(layers.Layer):
    expansion = 1

    # downsample 下采样函数
    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides,
                                   padding="SAME", use_bias=False)
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1,
                                   padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        # -----------------------------------------
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x

    # 瓶颈类，瓶颈两头大中间小，大小区分在深度，输入深度大（一头大），传入之后被压缩（中间小），输出被扩展（另一头大）


class Bottleneck(layers.Layer):
    expansion = 4

    # init好层的操作，这里均基于resnet-50，101，152的结构设计，这三层结构一致，只是这三层的组合数量不同
    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, use_bias=False, name="conv1")  # 默认步长1
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")
        # -----------------------------------------
        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, use_bias=False,
                                   strides=strides, padding="SAME", name="conv2")  # 从网络结构看，步长为2
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv2/BatchNorm")
        # -----------------------------------------
        self.conv3 = layers.Conv2D(out_channel * self.expansion, kernel_size=1, use_bias=False, name="conv3")  # 默认步长1
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv3/BatchNorm")
        # -----------------------------------------
        self.downsample = downsample
        self.add = layers.Add()
        self.relu = layers.ReLU()
        
    def call(self, inputs, training=False):

        # 如果该层需要快捷连接，即需要下采样
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)
        x = self.bn1(inputs, training=training)
        x = self.relu(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.bn3(x, training=training)
        x = self.conv3(x)

        x = self.add([x, identity])
        x = self.relu(x)

        return x

def _make_layer(block, in_channel, channel, block_num, name, strides=1):
    downsample = None
    # strides！=1表示输入会被降维，即输，或者输入in_channel和该层卷积最终输出深度channel*block.expansion不相等，则需要快捷连接
    if strides != 1 or in_channel != channel * block.expansion:
        downsample = Sequential([
            layers.Conv2D(channel * block.expansion, kernel_size=1, strides=strides,
                          use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")

    layers_list = []
    # 虚线残差结构，block的第一层卷积，名为unit_1
    layers_list.append(block(channel, downsample=downsample, strides=strides, name="unit_1"))
    # 实线残差结构
    for index in range(1, block_num):
        layers_list.append(block(channel, name="unit_" + str(index + 1)))

    return Sequential(layers_list, name=name)

def _resnet(block, blocks_num, im_width=224, im_height=224, num_classes=1000, include_top=True):
    # tensorflow中的tensor通道排序是NHWC
    # (None, 224, 224, 3)
    input_image = layers.Input(shape=(im_height, im_width, 3), dtype="float32")
    # 第一层卷积conv1
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2,
                      padding="SAME", use_bias=False, name="conv1")(input_image)
    # 第一层BN
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)
    # 第一层relu
    x = layers.ReLU()(x)

    # 第二层输入前最大池化下采样，高宽减半
    x = layers.MaxPool2D(pool_size=3, strides=2, padding="SAME")(x)

    # x.shape对应上一层输出特征矩阵的shpae，值为[batch,height,weight,channel]
    # 这里4个_make_layer对应论文网络结构中的conv2_x,conv3_x,conv4_x,conv5_x
    x = _make_layer(block, x.shape[-1], 64, blocks_num[0], name="block1")(x)
    x = _make_layer(block, x.shape[-1], 128, blocks_num[1], strides=2, name="block2")(x)
    x = _make_layer(block, x.shape[-1], 256, blocks_num[2], strides=2, name="block3")(x)
    x = _make_layer(block, x.shape[-1], 512, blocks_num[3], strides=2, name="block4")(x)

    # 顶层网络构建，即全连接层和max_pool层
    if include_top:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = layers.Dense(num_classes, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        predict = x

    model = Model(inputs=input_image, outputs=predict)
    
    return model

def build_model(class_num, image_size):
    # pretrained_backbone = tf.keras.applications.resnet50.ResNet50(
        # include_top=False
        # , weights='imagenet'
        # , pooling='avg'
    # )
    
    # pretrained_backbone.trainable = False
    
    pretrained_backbone = _resnet(BasicBlock, [3, 4, 6, 3], include_top=False)
    pretrained_backbone.summary()
    
    inputs = tf.keras.Input( shape=(*image_size, 3) ) 
    x = pretrained_backbone(inputs)
    x = tf.keras.layers.Dense(class_num, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model