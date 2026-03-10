import argparse
import os
import warnings
import cv2
import torch
import numpy as np
from tqdm import tqdm
from threading import Thread
from skimage import color, exposure, filters, util

visualize = False

warnings.filterwarnings("ignore")


def process_clip(event_path, normal_path):
    torch.cuda.set_device(device=0)
    event_data = torch.load(event_path)

    normal_cap = cv2.VideoCapture(normal_path)

    orb = cv2.ORB_create(nfeatures=2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    for i in tqdm(
        range(0, 5000, 50),
        ncols=86,
        mininterval=0.75,
        leave=False,
    ):
        for j in range(2):
            normal_cap.set(1, i + 10 * j)
            img_left2 = (
                event_data[i + 10 * j + 1].to_dense()
                + event_data[i + 10 * j + 2].to_dense()
                + event_data[i + 10 * j + 3].to_dense()
                + event_data[i + 10 * j + 4].to_dense()
            )  # Read the Event frame
            _, img_right = normal_cap.read()  # Read the Normal frame
            _ = normal_cap.grab()
            _ = normal_cap.grab()
            _ = normal_cap.grab()

            _, img_right2 = normal_cap.read()  # Read the Normal frame

            img_left = filters.gaussian(img_left2, 0.5)

            img_right = filters.gaussian(
                util.compare_images(
                    color.rgb2gray(img_right), color.rgb2gray(img_right2), method="diff"
                ),
                0.6,
            )

            img_left = exposure.rescale_intensity(img_left)
            img_right = exposure.rescale_intensity(img_right)

            # Convert to uint8 for ORB
            img_left_u8 = (img_left * 255).astype(np.uint8)
            img_right_u8 = (img_right * 255).astype(np.uint8)

            kp1, des1 = orb.detectAndCompute(img_left_u8, None)
            kp2, des2 = orb.detectAndCompute(img_right_u8, None)

            if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
                continue

            try:
                raw_matches = bf.knnMatch(des1, des2, k=2)

                # Lowe's ratio test
                good = [m for m, n in raw_matches if m.distance < 0.85 * n.distance]

                if len(good) < 4:
                    continue

                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=3.0)

                if H is not None and mask is not None and mask.sum() >= 4:
                    normal_cap.release()
                    tqdm.write("model found")
                    return H

            except Exception:
                continue

    normal_cap.release()
    return None


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
                if line.strip().split()[0] in str(("0", "2", "5", "7"))
                for value in line.split()[0:5]
            ]
            target_tensor = torch.tensor(target_data, dtype=torch.float16).view(-1, 5)
            target_tensors[int(filename[2].rsplit(".", maxsplit=1)[0]) - 1] = target_tensor
    return target_tensors


def process_clip_sequence(event_folder, normal_folder, start_clip, end_clip, results):
    matrices = []
    for i in range(start_clip, end_clip + 1):
        H = process_clip(f"{event_folder}event_frames_{i}.pt", f"{normal_folder}_{i}.mp4")
        if H is not None:
            matrices.append(H)

    if len(matrices) > 0:
        results.append(np.mean(matrices, axis=0))


def calc_H_matrix(input_dir, start_clip, end_clip, save_vid=False):
    matrices = []

    clip_per_thread = (end_clip - start_clip) // 3
    if not visualize:

        t1 = Thread(
            target=process_clip_sequence,
            args=[
                f"{input_dir}events/",
                f"{input_dir}normal/",
                start_clip + clip_per_thread * 2 + 1,
                end_clip,
                matrices,
            ],
        )
        t1.start()

        t2 = Thread(
            target=process_clip_sequence,
            args=[
                f"{input_dir}events/",
                f"{input_dir}normal/",
                start_clip + clip_per_thread * 1 + 1,
                start_clip + clip_per_thread * 2,
                matrices,
            ],
        )
        t2.start()

    process_clip_sequence(
        f"{input_dir}events/",
        f"{input_dir}normal/",
        start_clip,
        start_clip + clip_per_thread * 1,
        matrices,
    )

    if not visualize:
        t1.join()
        t2.join()

    if len(matrices) > 0:
        H = np.mean(matrices, axis=0)

        print("Homography matrix:")
        print(repr(H))
        np.save(f"{input_dir}/homography-matrix.npy", H)

        if save_vid:
            event_data = torch.load(f"{input_dir}events/event_frames_0.pt")

            normal_cap = cv2.VideoCapture(f"{input_dir}normal/_0.mp4")

            target_tensors = get_targets(f"{input_dir}track/labels_new/", 5400, 0)

            H_inv = np.linalg.inv(H)

            out = cv2.VideoWriter(
                filename=f"{input_dir}homography-vis.mp4",
                fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
                fps=90,
                frameSize=(640 + 736, 480),
                isColor=False,
            )

            for frame_idx in tqdm(
                range(int(len(event_data)) // 2),
                desc="annotating frames",
                ncols=86,
                mininterval=0.25,
            ):
                frame = event_data[frame_idx].to_dense() * 255
                ret, img_right = normal_cap.read()
                combined_frame = np.concatenate(
                    (
                        frame,
                        np.concatenate(
                            (cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY), np.full((20, 736), 255))
                        ),
                    ),
                    axis=1,
                )
                if target_tensors[frame_idx] is None:
                    continue
                for t in range(len(target_tensors[frame_idx])):
                    center_x, center_y = target_tensors[frame_idx][t][1:3].numpy() * [736, 460]
                    w, h = target_tensors[frame_idx][t][3:].numpy() * [736, 460]

                    x_min, y_min = np.array([center_x, center_y]) - [w / 2, h / 2]
                    x_max, y_max = np.array([center_x, center_y]) + [w / 2, h / 2]

                    # Transform RGB bounding box corners to event frame space
                    corners = np.float32([[x_min, y_min], [x_max, y_max]]).reshape(-1, 1, 2)
                    transformed = cv2.perspectiveTransform(corners, H_inv).reshape(-1, 2)
                    ex_min, ey_min = transformed[0]
                    ex_max, ey_max = transformed[1]

                    # Event frame box
                    cv2.rectangle(
                        img=combined_frame,
                        pt1=(int(ex_min), int(ey_min)),
                        pt2=(int(ex_max), int(ey_max)),
                        color=(255, 0, 0),
                        thickness=2,
                    )

                    # Normal frame box
                    cv2.rectangle(
                        img=combined_frame,
                        pt1=(int(x_min + 640), int(y_min)),
                        pt2=(int(x_max + 640), int(y_max)),
                        color=(255, 0, 0),
                        thickness=2,
                    )

                out.write(np.uint8(combined_frame))

            out.release()
            normal_cap.release()


parser = argparse.ArgumentParser(
    description="A script that calculates the homography matrix between event- and normal footage",
    usage="%(prog)s <path/to/data/> [options]",
)

parser.add_argument(
    "input_dir", help="The path to the directory containing the event, normal, and tracking data"
)

parser.add_argument(
    "-s",
    "--start-clip",
    default=0,
    type=int,
    help="The clip to start calculating the homography matrix from (default: %(default)s)",
)
parser.add_argument(
    "-e",
    "--end-clip",
    default=1,
    type=int,
    help="The clip to stop calculating the homography matrix at (default: %(default)s)",
)
parser.add_argument(
    "--save-vid",
    action="store_true",
    default=False,
    help="Can be set to also generate .mp4 files of the found homography matrix (default: %(default)s)",
)

args = parser.parse_args()

calc_H_matrix(args.input_dir, args.start_clip, args.end_clip, args.save_vid)
