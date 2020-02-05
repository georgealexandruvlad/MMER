EMOTION_DICT = {
    "Ne": "neutral",
    "Sa": "sadness",
    "Su": "surprise",
    "Fe": "fear",
    "An": "anger",
    "Di": "disgust",
    "Ha": 'happiness'
}

EMOTION_DICT_PRETRAIN = {
    "NE": "neutral",
    "SA": "sadness",
    "SU": "surprise",
    "AF": "fear",
    "AN": "anger",
    "DI": "disgust",
    "HA": 'happiness'
}

EMOTION_LABELS = {v: i for i, v in enumerate(sorted(EMOTION_DICT.values()))}
# audio padding
MAX_PAD_LEN = 550

config = {
    'epochs': 1000,
    'metric': 'val_categorical_accuracy',
    'mode': 'max',
    'train_batch_size': 8,
    'validation_batch_size': 8,
    'patience': 30,
    'model': 'Model_final'
}
