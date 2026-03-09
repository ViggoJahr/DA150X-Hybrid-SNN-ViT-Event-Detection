import argparse
from threading import Thread
import time
import cv2
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale
import event_streamer_c

# Removing hard-coded cuda_device - allowing for argparse.
# cuda_device = 0
decay_rate = 0.0002
frame_width = 640
frame_height = 480

decay = False  # set true to add decay to the input events

fps = 90


def pretty_time(seconds):
    if not seconds:
        return f"0s"
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    measures = (
        (hours, "h"),
        (minutes, "m"),
        (seconds, "s"),
    )
    return " ".join([f"{count}{unit}" for (count, unit) in measures if count])


def bin_events(fpath, output_dir, clip_length, num_of_clips, save_vids, gpu_id):
    torch.cuda.set_device(device=gpu_id)

    read_from = 239
    last_time_high = 0
    event_buffer, read_from, last_time_high = event_streamer_c.c_fill_event_buffer(
        f"{fpath}/events.raw", 20_000, read_from, last_time_high
    )
    df_timestamps = pd.read_csv(f"{fpath}/timestamps.csv")

    time_windows0 = df_timestamps.iloc[:, 0].to_numpy()
    time_windows10 = time_windows0 - time_windows0[0]

    total_runtime = 0
    event_idx = 0
    last_time_window = 0

    curr_evt = event_buffer[event_idx]
    event_idx += 1

    save_thread = None

    for clip_nr in range(num_of_clips):
        frames = []

        start_time = time.time()
        with tqdm(
            range(clip_length * fps),
            ncols=86,
            desc=f"processing clip {clip_nr+1}/{num_of_clips}",
            leave=False,
            mininterval=0.25,
            miniters=50,
            unit=" frames",
        ) as t_proc:
            for i in t_proc:
                start = time_windows10[last_time_window]
                end = time_windows10[last_time_window + 1]
                last_time_window += 1

                frame = np.zeros((frame_height, frame_width))
                while curr_evt and (curr_evt.timestamp - time_windows0[0]) < end:
                    decay_multiplier = 1
                    if decay:  # add decay rate if desired
                        time_since_start = curr_evt.timestamp - start
                        decay_multiplier = np.exp(-(time_since_start * decay_rate))

                    if (
                        curr_evt.x >= 0
                        and curr_evt.x < 640
                        and curr_evt.y >= 0
                        and curr_evt.y < 480
                    ):
                        frame[curr_evt.y, curr_evt.x] = curr_evt.polarity * decay_multiplier

                    if event_idx >= len(event_buffer):
                        event_buffer, read_from, last_time_high = (
                            event_streamer_c.c_fill_event_buffer(
                                f"{fpath}/events.raw", 20_000, read_from, last_time_high
                            )
                        )
                        event_idx = 0
                    curr_evt = event_buffer[event_idx]
                    event_idx += 1

                if not curr_evt:
                    print("Stream end reached. current clip processing aborted...")
                    exit(0)

                frames.append(torch.tensor(frame).to_sparse())

        filename = f"{output_dir}event_frames_{clip_nr}.pt"
        torch.save(torch.stack(frames), filename)

        proc_time = time.time() - start_time
        if save_vids:
            if save_thread and save_thread.is_alive():
                save_thread.join()
            save_thread = Thread(
                target=save_clip,
                args=[frames.copy(), output_dir, clip_nr, num_of_clips, fps, proc_time],
            )
            save_thread.start()
        else:
            tqdm.write(
                f"Clip {clip_nr+1}/{num_of_clips} [\x1b[92m\u2713\x1b[0m] Finished in \x1b[1m{pretty_time(proc_time)}\x1b[22m\n"
            )
        total_runtime += proc_time

    if save_thread and save_thread.is_alive():
        save_thread.join()

    print("==========================================")
    print(f"Processing finished in \x1b[1m{pretty_time(total_runtime)}\x1b[22m")
    print(f"Data saved to \x1b[1m{output_dir}\x1b[22m")


def save_clip(frames, output_dir, current_clip, num_of_clips, fps, proc_time):
    start_time = time.time()
    out = cv2.VideoWriter(
        f"{output_dir}event_clip_{current_clip}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
        False,
    )

    shape = frames[0].shape
    for f in tqdm(
        frames,
        desc=f"saving clip {current_clip+1}/{num_of_clips}",
        ncols=86,
        mininterval=0.25,
        leave=False,
    ):
        image_scaled = minmax_scale(
            f.to_dense().numpy().ravel(), feature_range=(0, 255), copy=False
        ).reshape(shape)
        out.write(np.uint8(image_scaled))

    out.release()
    save_time = time.time() - start_time

    tqdm.write(
        f"\nClip {current_clip+1}/{num_of_clips} [\x1b[92m\u2713\x1b[0m] Finished in \x1b[1m{pretty_time(proc_time+save_time)}\x1b[22m\n"
    )


parser = argparse.ArgumentParser(
    description="A script that reads an EVT2 file as an Event Stream and bins events into frames of given size",
    usage="%(prog)s <path/to/event_file> <path/to/output_dir/> [options]",
)

parser.add_argument("filename", help="The path to the file containing the EVT2 RAW event data")
parser.add_argument(
    "output_dir", help="The path to the directory where the processed data should be saved"
)
parser.add_argument(
    "-n",
    "--clips_count",
    default=1,
    type=int,
    help="The number of videos/clips to cut the event frames into (default: %(default)ss)",
)
parser.add_argument(
    "-l",
    "--length",
    default=60,
    type=int,
    help="The desired length of the clips in seconds (default: %(default)ss)",
)
parser.add_argument(
    "--save-vid",
    action="store_true",
    default=False,
    help="Can be set to also generate .mp4 files of the processed event frames (default: %(default)s)",
)
parser.add_argument(
    "--gpu",
    type=int,
    default=2,
    help="The CUDA device ID to use (default: %(default)s)",
)

args = parser.parse_args()

bin_events(args.filename, args.output_dir, args.length, args.clips_count, args.save_vid, args.gpu)
