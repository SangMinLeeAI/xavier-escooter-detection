import numpy as np
from matplotlib import pyplot as plt, patches


def plot_image(
        img: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        class_label: np.ndarray,
        save_path: object = None,
) -> object:
    """
    Function that draws the BBoxes, scores, and labels on the image.

    inputs:
      img: input-image as numpy.array (shape: [H, W, C])
      boxes: list of bounding boxes (Format [N, 4] => N times [xmin, ymin, xmax, ymax])
      scores: list of conf-scores (Format [N] => N times confidence-score between 0 and 1)
      labels: list of class-prediction (Format [N] => N times an number between 0 and _num_classes-1)
      dataset: list of all classes e.g. ["background", "class1", "class2", ..., "classN"] => Format [N_classes]
    """

    cmap = plt.get_cmap("tab20b")
    class_labels = np.array(class_label)
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    height, width, _ = img.shape
    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(16, 8))
    # Display the image
    ax.imshow(img)
    for i, box in enumerate(boxes):
        class_pred = labels[i]
        conf = scores[i]
        width = box[2] - box[0]
        height = box[3] - box[1]
        rect = patches.Rectangle(
            (box[0], box[1]),
            width,
            height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            box[0],
            box[1],
            s=class_labels[int(class_pred)] + " " + str(int(100 * conf)) + "%",
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # Used to save inference phase results
    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
