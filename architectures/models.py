import tensorflow as tf
from paths import MODEL_PATH
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, ZeroPadding2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM, Conv2D, MaxPool2D, MaxPooling2D, Concatenate
from tensorflow.keras.layers import TimeDistributed, Flatten, Dropout, Activation
import os


def get_model(input_shape: dict, name='VGG19'):
    if name == 'VGG16_tf':
        return VGG16_tf(input_shape)
    elif name == 'VGG16_unboxed':
        return VGG16_unboxed(input_shape)
    elif name == 'VGG16_pretrain':
        return VGG16_pretrain(input_shape)
    elif name == 'Model_final':
        return final_model(input_shape)


def final_model(input_shape: dict):
    video_frames = input_shape['video_frames']
    video_rows = input_shape['video_height']
    video_columns = input_shape['video_width']
    video_channels = input_shape['video_channels']
    audio_num_rows = input_shape['audio_num_rows']
    audio_num_columns = input_shape['audio_num_columns']
    audio_num_channels = input_shape['audio_num_channels']
    motion_frames = input_shape['motion_frames']
    motion_features = input_shape['motion_features']
    classes = input_shape['classes']

    video_input = Input(shape=(video_frames, video_rows, video_columns, video_channels))
    audio_input = Input(shape=(audio_num_rows, audio_num_columns, audio_num_channels))
    motion_input = Input(shape=(motion_frames, motion_features))

    # take for video
    video_masked = tf.keras.layers.Masking(mask_value=0)(video_input)
    vgg16 = VGG16(input_shape=(video_rows, video_columns, video_channels), weights="imagenet", include_top=False)
    vgg16_out = GlobalAveragePooling2D()(vgg16.output)
    vgg16_model = Model(inputs=vgg16.input, outputs=vgg16_out)
    vgg16_model.trainable = False

    encoded_frames = TimeDistributed(vgg16_model)(video_masked)
    encoded_sequence = LSTM(512)(encoded_frames)
    dense_video = Dense(256, activation="relu")(encoded_sequence)

    # sequences, _, lstm_output = LSTM(512, return_sequences=True, return_state=True,
    #                                  recurrent_initializer='glorot_uniform')(encoded_frames)
    # attention_layer = BahdanauAttention(10)
    # attention_result, attention_weights = attention_layer(lstm_output, sequences)
    # dense_video = Dense(256, activation="relu")(attention_result)

    # take for audio
    audio_conv1 = Conv2D(filters=16, kernel_size=2, activation='relu')(audio_input)
    audio_maxpool1 = MaxPooling2D(pool_size=2)(audio_conv1)
    audio_dp1 = Dropout(0.2)(audio_maxpool1)

    audio_conv2 = Conv2D(filters=32, kernel_size=2, activation='relu')(audio_dp1)
    audio_maxpool2 = MaxPooling2D(pool_size=2)(audio_conv2)
    audio_dp2 = Dropout(0.2)(audio_maxpool2)

    audio_conv3 = Conv2D(filters=64, kernel_size=2, activation='relu')(audio_dp2)
    audio_maxpool3 = MaxPooling2D(pool_size=2)(audio_conv3)
    audio_dp3 = Dropout(0.2)(audio_maxpool3)

    audio_conv4 = Conv2D(filters=128, kernel_size=2, activation='relu')(audio_dp3)
    audio_maxpool4 = MaxPooling2D(pool_size=2)(audio_conv4)
    audio_dp4 = Dropout(0.2)(audio_maxpool4)
    audio_global_avg = GlobalAveragePooling2D()(audio_dp4)

    # take for motion
    motion_LSTM = LSTM(128, return_sequences=False)(motion_input)
    motion_dense1 = Dense(64)(motion_LSTM)

    concat = Concatenate()([dense_video, audio_global_avg, motion_dense1])
    dp = Dropout(0.3)(concat)
    dense1 = Dense(128, activation='relu')(dp)
    output = Dense(classes, activation='softmax')(dense1)

    model = Model([video_input, audio_input, motion_input], output)
    return model


def final_model2(input_shape: dict):
    video_frames = input_shape['video_frames']
    video_rows = input_shape['video_height']
    video_columns = input_shape['video_width']
    video_channels = input_shape['video_channels']
    audio_num_rows = input_shape['audio_num_rows']
    audio_num_columns = input_shape['audio_num_columns']
    audio_num_channels = input_shape['audio_num_channels']
    motion_frames = input_shape['motion_frames']
    motion_features = input_shape['motion_features']
    classes = input_shape['classes']

    video_input = Input(shape=(video_frames, video_rows, video_columns, video_channels))
    audio_input = Input(shape=(audio_num_rows, audio_num_columns, audio_num_channels))
    motion_input = Input(shape=(motion_frames, motion_features))

    # take for video
    video_masked = tf.keras.layers.Masking(mask_value=0)(video_input)
    vggface = VGG16_face(input_shape)
    out_layer = vggface.get_layer(name='conv2d_14')
    cnn_out = GlobalAveragePooling2D()(out_layer.output)
    cnn = Model(inputs=vggface.input, outputs=cnn_out)
    cnn.trainable = False

    encoded_frames = TimeDistributed(cnn)(video_masked)
    encoded_sequence = LSTM(1024, recurrent_dropout=0.2, dropout=0.5)(encoded_frames)
    dense_video = Dense(256, activation="relu")(encoded_sequence)

    # take for audio
    audio_conv1 = Conv2D(filters=16, kernel_size=2, activation='relu')(audio_input)
    audio_maxpool1 = MaxPooling2D(pool_size=2)(audio_conv1)
    audio_dp1 = Dropout(0.2)(audio_maxpool1)

    audio_conv2 = Conv2D(filters=32, kernel_size=2, activation='relu')(audio_dp1)
    audio_maxpool2 = MaxPooling2D(pool_size=2)(audio_conv2)
    audio_dp2 = Dropout(0.2)(audio_maxpool2)

    audio_conv3 = Conv2D(filters=64, kernel_size=2, activation='relu')(audio_dp2)
    audio_maxpool3 = MaxPooling2D(pool_size=2)(audio_conv3)
    audio_dp3 = Dropout(0.2)(audio_maxpool3)

    audio_conv4 = Conv2D(filters=128, kernel_size=2, activation='relu')(audio_dp3)
    audio_maxpool4 = MaxPooling2D(pool_size=2)(audio_conv4)
    audio_dp4 = Dropout(0.2)(audio_maxpool4)
    audio_global_avg = GlobalAveragePooling2D()(audio_dp4)

    # take for motion
    motion_LSTM = LSTM(128, return_sequences=False)(motion_input)
    motion_dense1 = Dense(64)(motion_LSTM)

    concat = Concatenate()([dense_video, audio_global_avg, motion_dense1])
    dp1 = Dropout(0.5)(concat)
    dense1 = Dense(128, activation='relu')(dp1)
    output = Dense(classes, activation='softmax')(dense1)

    model = Model([video_input, audio_input, motion_input], output)
    return model


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(query, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # the shape of the tensor before applying self.V is (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(
            self.W1(values) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


def VGG16_tf1(input_shape: dict):
    frames = input_shape['video_frames']
    rows = input_shape['video_height']
    columns = input_shape['video_width']
    channels = input_shape['video_channels']
    classes = input_shape['classes']

    video = Input(shape=(frames,
                         rows,
                         columns,
                         channels))
    video_masked = tf.keras.layers.Masking(mask_value=0)(video)
    cnn_base = VGG16(input_shape=(rows,
                                  columns,
                                  channels),
                     weights="imagenet",
                     include_top=False)
    cnn_out = GlobalAveragePooling2D()(cnn_base.output)
    cnn = Model(inputs=cnn_base.input, outputs=cnn_out)
    cnn.trainable = True
    for layer in cnn.layers[:-8]:
        layer.trainable = False

    encoded_frames = TimeDistributed(cnn)(video_masked)
    encoded_sequence = LSTM(512)(encoded_frames)
    hidden_layer = Dense(256, activation="relu")(encoded_sequence)
    outputs = Dense(classes, activation="softmax")(hidden_layer)
    model = Model([video], outputs)
    return model


def VGG16_tf(input_shape: dict):
    frames = input_shape['frames']
    rows = input_shape['height']
    columns = input_shape['width']
    channels = input_shape['channels']
    classes = input_shape['classes']

    video = Input(shape=(frames,
                         rows,
                         columns,
                         channels))
    video_masked = tf.keras.layers.Masking(mask_value=0)(video)
    vggface = VGG16_pretrain(input_shape)
    out_layer = vggface.get_layer(name='dense_1')
    cnn = Model(inputs=vggface.input, outputs=out_layer.output)
    cnn.load_weights(os.path.join(MODEL_PATH, 'Model_pretrained', 'checkpoint')).expect_partial()
    cnn.trainable = False

    encoded_frames = TimeDistributed(cnn)(video_masked)
    # encoded_sequence = LSTM(512, recurrent_dropout=0.2)(encoded_frames)
    encoded_sequence = LSTM(512, dropout=0.2)(encoded_frames)

    hidden_layer1 = Dense(256, activation="relu")(encoded_sequence)

    outputs = Dense(classes, activation="softmax")(hidden_layer1)
    model = Model([video], outputs)
    return model


def VGG16_pretrain(input_shape: dict):
    rows = input_shape['height']
    columns = input_shape['width']
    channels = input_shape['channels']
    classes = input_shape['classes']

    image = Input(shape=(rows,
                         columns,
                         channels))
    vggface = VGG16_face(input_shape)
    out_layer = vggface.get_layer(name='conv2d_14')
    cnn_out = GlobalAveragePooling2D()(out_layer.output)
    cnn = Model(inputs=vggface.input, outputs=cnn_out)
    cnn.trainable = False

    out = cnn(image)
    hidden_layer1 = Dense(1024)(out)
    dp1 = Dropout(0.5)(hidden_layer1)
    hidden_layer2 = Dense(512)(dp1)
    outputs = Dense(classes, activation="softmax")(hidden_layer2)
    model = Model([image], outputs)
    return model


def VGG16_face(input_shape: dict):
    rows = input_shape['video_height']
    columns = input_shape['video_width']
    channels = input_shape['video_channels']

    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(rows, columns, channels)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(512, (3, 3), activation='relu'))
    model.add(MaxPool2D((2, 2), strides=(2, 2)))

    model.add(Conv2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(2622, (1, 1)))
    model.load_weights(os.path.join(MODEL_PATH, 'vgg_face_base', 'checkpoint')).expect_partial()
    return model


def VGG16_unboxed(input_shape: dict):
    frames = input_shape['frames']
    rows = input_shape['height']
    columns = input_shape['width']
    channels = input_shape['channels']
    classes = input_shape['classes']

    video = Input(shape=(frames, rows, columns, channels))
    video_masked = tf.keras.layers.Masking(mask_value=0)(video)

    image = Input(shape=(rows, columns, channels))
    block1_conv1 = Conv2D(filters=64, kernel_size=(3, 3), padding="same",
                          activation="relu")(image)
    block1_conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(block1_conv1)
    block1_maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(block1_conv2)

    block2_conv1 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(block1_maxpool)
    block2_conv2 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(block2_conv1)
    block2_maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(block2_conv2)

    block3_conv1 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(block2_maxpool)
    block3_conv2 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(block3_conv1)
    block3_conv3 = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(block3_conv2)
    block3_maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(block3_conv3)

    block4_conv1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(block3_maxpool)
    block4_conv2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(block4_conv1)
    block4_conv3 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(block4_conv2)
    block4_maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(block4_conv3)

    block5_conv1 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(block4_maxpool)
    block5_conv2 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(block5_conv1)
    block5_conv3 = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(block5_conv2)
    block5_maxpool = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(block5_conv3)

    vgg16_out = GlobalAveragePooling2D()(block5_maxpool)
    vgg16_model = Model(inputs=image, outputs=vgg16_out)

    encoded_frames = TimeDistributed(vgg16_model)(video_masked)
    encoded_sequence = LSTM(512)(encoded_frames)
    hidden_layer = Dense(256, activation="relu")(encoded_sequence)
    outputs = Dense(classes, activation="softmax")(hidden_layer)
    model = Model([video], outputs)
    return model


if __name__ == '__main__':
    d = dict()
    d["batch"] = 2
    d["video_frames"] = 10
    d["video_height"] = 400
    d["video_width"] = 400
    d["video_channels"] = 3
    d["classes"] = 7
    d["audio_num_rows"] = 40
    d["audio_num_columns"] = 550
    d["audio_num_channels"] = 1
    d["motion_frames"] = 10
    d["motion_features"] = 20

    # model = VGG16_unboxed(d)
    # model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])
    # model.summary()
    #
    # import h5py
    #
    # f = h5py.File(os.path.join(DATA_PATH, 'vgg_face_weights.h5'), 'r')
    # print(list(f.keys()))

    model = VGG16_face(d)
    model.compile('sgd', 'categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    # model.save_weights(os.path.join(MODEL_PATH, 'vgg_face_base', 'checkpoint'))

    # saver = tf.train.Checkpoint()
    # model = VGG16_face(d)
    # # sess = tf.compat.v1.keras.backend.get_session()
    # save_path = saver.save('vgg_face_weights')
