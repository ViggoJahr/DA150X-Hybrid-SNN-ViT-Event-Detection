import argparse
import argparse
import cv2
import numpy as np
import torch
from tqdm import tqdm


rois = [(47 - 35, 137 - 10, 303 - 35, 393 - 10), (635 - 256, 137 - 10, 635, 393 - 10)]



def visuliaze_rois(input_dir):
    data = torch.load("data/event_tensors/event_frames_0.pt")

    out = cv2.VideoWriter(
        filename=f"{input_dir}roi-vis.mp4",
        fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
        fps=90,
        frameSize=(640, 480),
        isColor=True,
    )

    for i in tqdm(
        range(5400 // 2),
        ncols=86,
        position=0,
        leave=False,
        mininterval=0.25,
        miniters=50,
    ):
        frame = (torch.stack((data[i].to_dense(),) * 3, axis=-1) * 255).cpu().numpy()
        for roi in rois:
            cv2.rectangle(
                img=frame,
                pt1=(int(roi[0]), int(roi[1])),
                pt2=(int(roi[2]), int(roi[3])),
                color=(0, 0, 255),
                thickness=1,
            )
        out.write(np.uint8(frame))

    out.release()
    np.save(f"{input_dir}rois", rois)


parser = argparse.ArgumentParser(
    description="A script that visualizes the chossen rois over the event data",
    usage="%(prog)s <path/to/data/>",
)

parser.add_argument("input_dir", help="The path to the directory containing the event data")

parser.add_argument(
    "--gpu",
    type=int,
    default=2,
    help="The CUDA device ID to use (default: %(default)s)")

args = parser.parse_args()

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
visuliaze_rois(args.input_dir)
