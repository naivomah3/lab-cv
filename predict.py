import os, cv2
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from models.vgg_fcn8 import vgg_fcn8s
from helpers.data_generator import image_data_generator, image_data_builder
from helpers.data_slicing import get_rand_name


# Predict samples from a checkpoint saved model
def predict(frames_path=None, masks_path=None, model_path=None, n_classes=None, is_generator=False):
    predicted_masks = None
    test_frames = None
    test_masks = None

    # Get image fnames to predict
    fnames = os.listdir(frames_path)

    # Load pretrained model
    model = vgg_fcn8s(pretrained=True, model_path=model_path)

    if is_generator:
        test_image_generator = image_data_generator(frames_path=frames_path, masks_path=masks_path, fnames=fnames, n_classes=n_classes, batch_size=10)
        predicted_masks = model.predict_generator(test_image_generator, steps=15)
    else:
        test_frames, test_masks = image_data_builder(frames_path=frames_path, masks_path=masks_path, fnames=fnames)
        predicted_masks = model.predict(test_frames, batch_size=10)

    return test_frames, test_masks, predicted_masks


def color_mask(mask=None, n_classes=None):
    # if depth of mask == no. of classes, then consider the last depth
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    mask_chanels = np.zeros((mask.shape[0], mask.shape[1], 3)).astype("float")
    colors = sns.color_palette('bright', 10)
    for i in range(n_classes):
        mask_color = (mask == i)
        mask_chanels[:, :, 0] += (mask_color * (colors[i + 2][0]))
        mask_chanels[:, :, 1] += (mask_color * (colors[i + 2][1]))
        mask_chanels[:, :, 2] += (mask_color * (colors[i + 2][2]))

    return (mask_chanels)


def plot_prediction(frames=None, masks=None, predictions=None, frames_out_path=None, masks_out_path=None, predict_path=None):
    for i in range(50):
        rand_name = get_rand_name()
        print(np.unique(color_mask(masks[i])))
        fig = plt.figure(figsize=(14, 20))
        axs = fig.add_subplot(1, 3, 1)
        axs.set_title(f"Frame sample: ({frames[i].shape[0]}, {frames[i].shape[1]})")
        axs.imshow(frames[i] / 255.0)
        # Save frame as png
        cv2.imwrite(os.path.join(frames_out_path, rand_name + '.png'), frames[i])

        axs = fig.add_subplot(1, 3, 2)
        axs.set_title(f"Ground truth: ({masks[i].shape[0]}, {masks[i].shape[1]})")
        axs.imshow(color_mask(masks[i]))
        # Save mask as png
        cv2.imwrite(os.path.join(masks_out_path, rand_name + '.png'), color_mask(masks[i]) * 230)

        axs = fig.add_subplot(1, 3, 3)
        axs.set_title(f"Predicted mask: ({predictions[i].shape[0]}, {predictions[i].shape[1]})")
        axs.imshow(color_mask(predictions[i]))
        # Save predicted mask as png
        cv2.imwrite(os.path.join(predict_path, rand_name + '.png'), color_mask(predictions[i]) * 230)

        plt.show()


# Compute per class IoU and F1-score(Dice)
def evaluate(masks=None, predictions=None, n_classes=None):
    per_class_iou = dict()
    per_class_fscore = dict()
    per_class_accuracy = dict()
    per_class_recall = dict()
    per_class_precision = dict()
    for label in range(n_classes):
        true_positive = np.sum((masks == label) & (predictions == label))
        false_positive = np.sum((masks != label) & (predictions == label))
        false_negative = np.sum((masks == label) & (predictions != label))
        true_negative = np.sum((masks != label) & (predictions != label))
        per_class_precision[label] = true_positive / float(true_positive + false_positive)
        per_class_recall[label] = true_positive / float(true_positive + false_negative)
        per_class_iou[label] = true_positive / float(true_positive + false_positive + false_negative)
        per_class_fscore[label] = (2. * per_class_precision[label] * per_class_recall[label]) / float(
            per_class_precision[label] + per_class_recall[label])
        per_class_accuracy[label] = (true_positive + true_negative) / float(
            true_negative + true_positive + false_positive + false_negative)

    return per_class_iou, per_class_fscore, per_class_precision, per_class_recall, per_class_accuracy

# Just print IoU/dices/Precision/Recall/Accuracy
def print_scores(ious=None, dices=None, precision=None, recall=None, accuracy=None):
    metrics = set(ious) & set(dices) & set(accuracy) & set(recall) & set(precision)
    for c in metrics:
        print(
            f"class {c}({labels[c]}): Accuracy={accuracy[c]:.4f}, Precision={precision[c]:.4f}, Recall={recall[c]:.4f}, IoU={ious[c]:.4f}, F1-score={dices[c]:.4f},")

    avg_iou = np.mean([float(x) for x in ious.values()])
    avg_dice = np.mean([float(x) for x in dices.values()])
    avg_accuracy = np.mean([float(x) for x in accuracy.values()])
    avg_precision = np.mean([float(x) for x in precision.values()])
    avg_recall = np.mean([float(x) for x in recall.values()])

    print(
        f"Avg: Accuracy={avg_accuracy:.4f}, Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, IoU={avg_iou:.4f}, F1-score={avg_dice:.4f}")

# Load history from file
def load_hist(filename):
    # Load history from a pickle
    with open(filename, 'rb') as handle:
        hist = load(handle)
    return hist

# Plot training history from file
def plot_history(history_file):
    history = load_hist(history_file)
    fig = plt.figure(figsize=(14, 20))
    axs = fig.add_subplot(1, 2, 1)
    axs.set_title("FCN-8")
    axs.plot(history['loss'], 'b', label='Training loss')
    axs.plot(history['val_loss'], 'r', label='Validation loss')
    axs.legend()

    axs = fig.add_subplot(1, 2, 2)
    axs.set_title("U-Net")
    axs.plot(history['dice'], 'b', label='Training dice')
    axs.plot(history['val_dice'], 'r', label='Validation dice')
    axs.legend()

    plt.show()


#-------------------------------------------------------------------------------------------
# Validation path(frames+masks)
frames_path = 'dataset/test/frames'
masks_path = 'dataset/test/masks'
model_path = 'vgg_fcn8s.model'      # change to 'vgg_unet.model' to load U-Net model
history_path = 'segroad_fcn8_train_26_11_19.history'    # change to 'segroad_unet_train_26_11_19.history' to load U-Net history
labels = ['background', 'road', 'occluded_road']
n_classes = len(labels)

# Prediction
frames, masks, predicts = predict(frames_path=frames_path, masks_path=masks_path, model_path=model_path, n_classes=n_classes)

# OneHot masks and predicted mask upon probabilities
masks = np.argmax(masks, axis=3)
predicts = np.argmax(predicts, axis=3)

# Evaluate the model
ious, dices, precision, recall, accuracy = evaluate(masks=masks, predictions=predicts, n_classes=n_classes)

# Get scores
print_scores(ious, dices, precision, recall, accuracy)

# Plot history
plot_history(history_path)