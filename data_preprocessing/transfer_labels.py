import argparse
from concurrent.futures import thread
from math import exp
from pathlib import Path
from threading import Thread
import warnings
import cv2
import torch
import numpy as np
from tqdm import tqdm
from skimage import transform
import os

# from calc_homography_matrix import calc_H_matrix

""" 
1. Get the labels from yolo/result/labels
2. Map the coordinates with the event video

"""


# 0 - Person
# 1 - Bicycle
# 2 - Car
# 5 - Bus
# 7 - Truck

tracked_classes = [0, 2, 5, 7]

warnings.filterwarnings("ignore", message="__array_wrap__*")


def get_targets(directory, target_length, file_nr):
    target_tensors = np.empty(target_length, dtype=object)

    for filename in sorted(filter(lambda f: f.endswith(".txt"), os.listdir(directory))):
        with open(os.path.join(directory, filename)) as file:
            filename = filename.split("_")
            if int(filename[1]) != file_nr:
                continue
            target_data = [
                float(value)
                for line in file
                if line.strip().split()[0] in str(tracked_classes)
                for value in line.split()[0:5]
            ]
            target_tensor = torch.tensor(target_data, dtype=torch.float16).view(-1, 5)
            target_tensors[int(filename[2].rsplit(".", maxsplit=1)[0]) - 1] = target_tensor
    return target_tensors


def count_events(frame, min_max, class_type):
    y_min, y_max, x_min, x_max = min_max
    number_of_events = torch.sum(frame > 0)
    min_events_threshold = max(
        1, (0.002 if class_type == 0 else 0.01) * ((x_max - x_min) * (y_max - y_min))
    )  # Dynamic minimum based on bbox area
    if number_of_events < min_events_threshold:
        ratio = 0
    else:
        ratio = (number_of_events - min_events_threshold) / (
            (x_max - x_min) * (y_max - y_min) + 0.000000001
        )

    return ratio


def leaky_integrator(input_data, tau, v):
    v_next = v + (1 / tau) * (input_data - v)
    return v_next


def annotate_frame(frame, targets, overlays, out, roi, clip_maximum, visualize):
    x_min, y_min, x_max, y_max = roi
    small_frame_dim = 64
    G_overlays = {}

    for trck_class in tracked_classes:
        G_overlays[trck_class] = torch.zeros((small_frame_dim, small_frame_dim))

    tempframe = np.stack((frame.detach().clone().numpy(),) * 3, axis=-1) * 255
    for t in range(len(targets)):
        class_type = targets[t][0].item()

        center_x, center_y = targets[t][1] - [roi[0], roi[1]]
        w, h = targets[t][2]

        x_min, y_min = np.maximum(
            [0, 0], np.minimum([256, 256], np.array([center_x, center_y]) - [w / 2, h / 2])
        )

        x_max, y_max = np.maximum(
            [0, 0], np.minimum([256, 256], np.array([center_x, center_y]) + [w / 2, h / 2])
        )

        if class_type == 0:
            cv2.rectangle(
                img=tempframe,
                pt1=(int(max(0, x_min - 3)), int(max(0, y_min - 3))),
                pt2=(int(min(256, x_max + 3)), int(min(256, y_max + 3))),
                color=(0, 0, 255),
                thickness=1,
            )
        else:
            cv2.rectangle(
                img=tempframe,
                pt1=(int(x_min), int(y_min)),
                pt2=(int(x_max), int(y_max)),
                color=(0, 0, 255),
                thickness=1,
            )

        # Scale down the values to the 256x256 frame
        w_small = torch.tensor(((x_max - x_min) / 256) * 64)
        h_small = torch.tensor(((y_max - y_min) / 256) * 64)
        center_x_small = (center_x / 256) * 64
        center_y_small = (center_y / 256) * 64

        # Adjust sigma proportionally for the smaller frame
        sigma_x_small = w_small / (4 if class_type == 0 else 6)
        sigma_y_small = h_small / (4 if class_type == 0 else 6)

        if class_type == 0:

            events_factor = count_events(
                frame[
                    max(0, int(y_min) - 3) : max(0, int(y_max) - 3),
                    min(255, int(x_min) + 3) : min(255, int(x_max) + 3),
                ],
                (y_min, y_max, x_min, x_max),
                class_type,
            )
        else:
            events_factor = count_events(
                frame[
                    int(y_min) : int(y_max),
                    int(x_min) : int(x_max),
                ],
                (y_min, y_max, x_min, x_max),
                class_type,
            )

        # Create Gaussian mask on the smaller frame
        X, Y = np.meshgrid(
            np.linspace(0, small_frame_dim, small_frame_dim),
            np.linspace(0, small_frame_dim, small_frame_dim),
        )
        G = np.exp(
            -(
                (X - center_x_small) ** 2 / (2 * sigma_x_small**2)
                + (Y - center_y_small) ** 2 / (2 * sigma_y_small**2)
            )
        )
        G_normalized = G * events_factor

        G_overlays[class_type] = np.maximum(G_overlays[class_type], G_normalized)

        # Apply to the small overlay

    # if torch.max(G_overlays[class_type]) > clip_maximum:
    #     clip_maximum = torch.max(G_overlays[class_type])

    for trck_class in tracked_classes:
        overlays[trck_class] = leaky_integrator(
            G_overlays[trck_class].detach().clone(),
            tau=10,
            v=overlays[trck_class].detach().clone(),
        )
    x_min, y_min, x_max, y_max = roi
    if visualize:
        scaledLabels = []
        for trck_class in tracked_classes:
            temp = overlays[trck_class].detach().clone().numpy()
            temp = np.stack(((temp / max(np.max(temp), 0.00001)),) * 3, axis=-1) * 255

            scaledLabels.append(
                cv2.resize(
                    temp,
                    dsize=(256, 256),
                    interpolation=cv2.INTER_NEAREST,
                )
            )
            cv2.putText(
                scaledLabels[len(scaledLabels) - 1],
                f"class: {trck_class}",
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_4,
            )
        tempframe = np.concatenate(
            (
                tempframe,
                np.full((256, 20, 3), 255),
            ),
            axis=1,
        )
        for i, scaledLabel in enumerate(scaledLabels):
            if i == len(scaledLabels) - 1:
                tempframe = np.concatenate((tempframe, scaledLabel), axis=1)
            else:
                tempframe = np.concatenate(
                    (tempframe, scaledLabel, np.full((256, 20, 3), 255)), axis=1
                )
        out.write(np.uint8(tempframe))

    return frame.detach().clone(), overlays


H = None

rois = None


def process_roi(i, j, roi, start_clip, end_clip, input_dir, save_dir, visualize):
    print(
        f"Processing clip: {i-start_clip+1}/{end_clip-start_clip+1} | ROI: {roi} ({j+1}/{len(rois)})"
    )
    targets = get_targets(f"{input_dir}track/labels/", 5400, i)
    frames_tensor = []
    labels_tensor = {}
    for trck_class in tracked_classes:
        labels_tensor[trck_class] = []
    data = torch.load(f"{input_dir}events/event_frames_{i}.pt")
    small_frame_dim = 64
    clip_maximum = 0
    overlays = {}
    for trck_class in tracked_classes:
        overlays[trck_class] = torch.zeros((small_frame_dim, small_frame_dim))
    out = None
    if visualize:
        out = cv2.VideoWriter(
            filename=f"{save_dir}{i}-{j}-vis.mp4",
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=90,
            frameSize=(
                256 * (len(tracked_classes) + 1) + 20 * len(tracked_classes),
                256,
            ),
            isColor=True,
        )

    with tqdm(
        range(int(len(data))),
        desc="annotating frames",
        ncols=86,
        mininterval=0.25,
    ) as t:
        for frame_idx in t:
            frame = data[frame_idx].to_dense()[roi[1] : roi[3], roi[0] : roi[2]]
            warped = []

            t.set_description("annotating frames")
            if targets[frame_idx] == None:
                t.set_description(f"missing: {frame_idx}")
                res = frame.detach().clone()

            elif len(targets[frame_idx]) == 0:
                res = frame.detach().clone()

            else:
                for tar in range(len(targets[frame_idx])):
                    x, y = targets[frame_idx][tar][1:3] * torch.tensor([736, 460])
                    w, h = targets[frame_idx][tar][3:] * torch.tensor([736, 460])

                    x_min, y_min = np.array([x, y]) - [w / 2, h / 2]
                    x_max, y_max = np.array([x, y]) + [w / 2, h / 2]

                    y_min, x_min = H._apply_mat((y_min, x_min), H.inverse.params)[0]
                    y_max, x_max = H._apply_mat((y_max, x_max), H.inverse.params)[0]
                    w = x_max - x_min
                    h = y_max - y_min

                    transformed_coordinates = H._apply_mat((y, x), H.inverse.params)[0]
                    warped.append(
                        [
                            targets[frame_idx][tar][0],
                            np.flip(transformed_coordinates),
                            torch.tensor([w, h]),
                        ]
                    )

                res, overlays = annotate_frame(
                    frame, warped, overlays, out, roi, clip_maximum, visualize
                )

            frames_tensor.append(res)
            for trck_class in tracked_classes:
                labels_tensor[trck_class].append(overlays[trck_class].detach().clone())

    print(f"\nSaving visualization to \x1b[1m{save_dir}{i}-{j}-vis.mp4\x1b[22m")
    if visualize:
        out.release()
    print(f"\nSaving data to \x1b[1m{save_dir}{i}-{j}.pt\x1b[22m")
    clip_data = [torch.stack(frames_tensor).to_sparse()]

    for trck_class in tracked_classes:
        clip_data.append(torch.stack(labels_tensor[trck_class]).to_sparse())

    torch.save(
        clip_data,
        f"{save_dir}{i}-{j}.pt",
    )


def stack_via_numpy1(tensor_list):
    return torch.from_numpy(np.array([ten.numpy() for ten in tensor_list]))


def generate_labels(input_dir, save_dir, start_clip, end_clip, only_roi, visualize):
    global H, rois
    threads = [None, None]

    rois = np.load(f"{input_dir}rois.npy")

    H = transform.SimilarityTransform()
    H.params = np.load(f"{input_dir}homography-matrix.npy")  # calc_H_matrix(input_dir, 4, 29, True)
    save_dir = f"{save_dir}{input_dir.split('/')[-4]}-{input_dir.split('/')[-2]}/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for i in range(start_clip, end_clip + 1):
        for j, roi in enumerate(rois):
            if only_roi and j != only_roi - 1:
                continue
            process_roi(i, j, roi, start_clip, end_clip, input_dir, save_dir, visualize)
        #     threads[j] = Thread(
        #         target=process_roi,
        #         args=[i, j, clip_idx, roi, start_clip, end_clip, input_dir, save_dir, visualize],
        #     )
        #     threads[j].start()
        # for j in range(len(threads)):
        #     threads[j].join()


parser = argparse.ArgumentParser(
    description="A script that creates the training dataset",
    usage="%(prog)s <path/to/data/> <path/to/output/> [options]",
)

parser.add_argument("input_dir", help="The path to the directory containing the event data")
parser.add_argument(
    "output_dir", help="The path to the directory where the generated data should be saved"
)

parser.add_argument(
    "-s",
    "--start-clip",
    required=True,
    type=int,
    help="The clip to start generating training data from (Required)",
)
parser.add_argument(
    "-e",
    "--end-clip",
    type=int,
    required=True,
    help="The clip to stop generating training data at (Required)",
)

parser.add_argument(
    "--roi",
    type=int,
    help="Choose specific ROI to only generate data for",
)

parser.add_argument(
    "--save-vid",
    action="store_true",
    default=False,
    help="Can be set to also generate .mp4 files of the generated training data (default: %(default)s)",
)

parser.add_argument(
    "--gpu",
    type=int,
    default=2,
    help="The CUDA device ID to use (default: %(default)s)",
)



args = parser.parse_args()

# Set cuda_device
torch.cuda.set_device(args.gpu)

generate_labels(
    args.input_dir, args.output_dir, args.start_clip, args.end_clip, args.roi, args.save_vidm
)

# w35/box1/1-09-04
# w38/box3/3-09-27/
# w31/box2/2-08-01/
