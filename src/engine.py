import random, string
import numpy as np

def get_rand_name(size=30, chars=string.ascii_letters + string.digits):
    '''
    Generate Random filename
    rtype: a random string of length 30
    '''
    return ''.join(random.choice(chars) for x in range(size))

def unison_shuffled(a, b):
    '''
    a & b: 2 numpy arrays
    '''
    if len(a) != len(b):
        raise Exception('Dimensions do not match')
    p = np.random.permutation(len(a))
    return a[p], b[p]


def color_img(frame, n_classes):
    '''
    Pixel wise coloring of each class label
    frame: 2-d matrix of all respective classes
    '''
    # create channels
    mask_channels = np.zeros((frame.shape[0], frame.shape[1], 3)).astype("float")
    # Create color palette
    # black =  [  0,   0,   0] > Background
    # red =    [209,  11,  40] > Road
    # blue =   [  0,  38, 230] > Occlusion
    # green =  [ 12, 188,  64] > Vegetation
    # violet = [222,   0, 292] > etc...
    # yellow = [245, 255,  82] > etc...
    R = [0, 208,   0,  12, 222, 245]
    G = [0,  11,  38, 188,   0, 255]
    B = [0,  40, 230,  64, 292,  82]

    if n_classes == 2:
        # Color labels for binary classification
        mask_channels *= 255
    else:
        # Color labels for multi label classification
        # Assuming the first label is Background and be colored with Black(0,0,0)
        for i in range(n_classes):
            mask_color = (frame == i)
            mask_channels[:, :, 0] += (mask_color * B[i])
            mask_channels[:, :, 1] += (mask_color * G[i])
            mask_channels[:, :, 2] += (mask_color * R[i])

    return (mask_channels)

# Compute per class IoU and F1-score(Dice)
def evaluate(masks=None, predictions=None, n_classes=None, smooth=1e-15):
    per_class_iou = dict()
    per_class_fscore = dict()
    per_class_accuracy = dict()
    per_class_recall = dict()
    per_class_precision = dict()
    # for label in (set(np.unique(masks)) | set(np.unique(predictions))):
    for label in range(n_classes):
        true_positive = np.sum((masks == label) & (predictions == label))
        false_positive = np.sum((masks != label) & (predictions == label))
        false_negative = np.sum((masks == label) & (predictions != label))
        true_negative = np.sum((masks != label) & (predictions != label))

        per_class_precision[label] = (true_positive + smooth) / float(true_positive + false_positive + smooth)
        per_class_recall[label] = (true_positive + smooth) / float(true_positive + false_negative + smooth)
        per_class_iou[label] = (true_positive + smooth) / float(true_positive + false_positive + false_negative + smooth)
        per_class_fscore[label] = (2. * per_class_precision[label] * per_class_recall[label] + smooth) / float(per_class_precision[label] + per_class_recall[label] + smooth)
        per_class_accuracy[label] = (true_positive + true_negative + smooth) / float(true_negative + true_positive + false_positive + false_negative + smooth)

    return per_class_iou, per_class_fscore, per_class_precision, per_class_recall, per_class_accuracy

# Just print IoU/dices/Precision/Recall/Accuracy
def print_scores(ious=None, dices=None, precision=None, recall=None, accuracy=None, labels=None):
    metrics = set(ious) & set(dices) & set(accuracy) & set(recall) & set(precision)
    # Per class evaluation
    for (i, _) in enumerate(labels):
        print(f"\nclass {i}({labels[i]}): \n"
              f"Accuracy={accuracy[i]:.4f}, "
              f"Precision={precision[i]:.4f}, "
              f"Recall={recall[i]:.4f}, "
              f"IoU={ious[i]:.4f}, "
              f"F1-score={dices[i]:.4f},")

    avg_iou = np.mean([float(x) for x in ious.values()])
    avg_dice = np.mean([float(x) for x in dices.values()])
    avg_accuracy = np.mean([float(x) for x in accuracy.values()])
    avg_precision = np.mean([float(x) for x in precision.values()])
    avg_recall = np.mean([float(x) for x in recall.values()])
    # Average evaluation
    print(f"\nAvg: \n"
          f"Accuracy={avg_accuracy:.4f}, "
          f"Precision={avg_precision:.4f}, "
          f"Recall={avg_recall:.4f}, "
          f"IoU={avg_iou:.4f}, "
          f"F1-score={avg_dice:.4f}")