from functools import partial
from matplotlib import animation
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from norse.torch import LILinearCell
from norse.torch.module.lif import LIFCell, LIFParameters
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import date
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse

# Set device
parser = argparse.ArgumentParser(description="Test traffic monitoring SNN model")
parser.add_argument("--gpu", type=int, default=2, help="The CUDA device ID to use (default: %(default)s)")
# One could also add an argument for the model here later.
args = parser.parse_args()

sequence_length, overlap, batch_size = 60, 25, 24
torch.cuda.set_device(args.gpu)

num_inputs = 256 * 256
num_outputs = 64 * 64  # 4096
# tau_list = [120, 140, 160, 180, 200] #For trying different taus
layer_nr = 850
w_decay = 1e-4  # 6.5e-3
lr = 1e-4
print_image = (
    False  # Set true to save current image for each loop, used to see progress during training
)
loss_function = nn.MSELoss()
tau_mem = 180


class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=7, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(8)
        self.lif1 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.conv2 = nn.Conv2d(8, 10, kernel_size=5, stride=2, padding=0)
        self.bn2 = nn.BatchNorm2d(10)
        self.lif2 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.conv3 = nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(10)
        self.lif3 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.lif4 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
        self.lif5 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
        self.lif6 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))
        self.lif7 = LIFCell(p=LIFParameters(tau_mem_inv=tau_mem))

        self.fcperson1 = nn.Linear(4000, layer_nr)
        self.fcperson2 = nn.Linear(layer_nr, layer_nr)
        self.lifperson = LILinearCell(layer_nr, 4096)

        self.fccar1 = nn.Linear(4000, layer_nr)
        self.fccar2 = nn.Linear(layer_nr, layer_nr)
        self.lifcar = LILinearCell(layer_nr, 4096)

        self.fcbus1 = nn.Linear(4000, layer_nr)
        self.fcbus2 = nn.Linear(layer_nr, layer_nr)
        self.lifbus = LILinearCell(layer_nr, 4096)

        self.fctruck1 = nn.Linear(4000, layer_nr)
        self.fctruck2 = nn.Linear(layer_nr, layer_nr)
        self.liftruck = LILinearCell(layer_nr, 4096)

        self.maxpool = nn.MaxPool2d(2, 2)
        # self.avgpool = nn.AvgPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.35)
        self.dropout2 = nn.Dropout(p=0.45)

    def forward(self, x, mem_states):
        batch_size, C, W, H = x.shape
        x = (x != 0).float()  # Ensure binary data for this case

        (
            mem1,
            mem2,
            mem3,
            mem5_1,
            mem5_2,
            mem6_1,
            mem6_2,
            mem7_1,
            mem7_2,
            mem8_1,
            mem8_2,
        ) = mem_states

        v1 = self.bn1(self.conv1(x))
        spk1, mem1 = self.lif1(v1, mem1)

        v2 = self.dropout1(self.bn2(self.conv2(self.maxpool(spk1))))
        # v2 = self.dropout(self.bn2(self.conv2(spk1)))
        spk2, mem2 = self.lif2(v2, mem2)

        v3 = self.dropout1(self.bn3(self.conv3(spk2)))
        # v3 = self.dropout1(self.maxpool(self.bn3(self.conv3(spk2))))
        spk3, mem3 = self.lif3(v3, mem3)

        spk3_flat = spk3.view(batch_size, -1)

        v5 = self.dropout2(self.fcperson1(spk3_flat))
        spk5_1, mem5_1 = self.lif4(v5, mem5_1)
        v5 = self.dropout2(self.fcperson2(spk5_1))
        spk5_2, mem5_2 = self.lifperson(v5, mem5_2)

        v6 = self.dropout2(self.fccar1(spk3_flat))
        spk6_1, mem6_1 = self.lif5(v6, mem6_1)
        v6 = self.dropout2(self.fccar2(spk6_1))
        spk6_2, mem6_2 = self.lifcar(v6, mem6_2)

        v7 = self.dropout2(self.fcbus1(spk3_flat))
        spk7_1, mem7_1 = self.lif6(v7, mem7_1)
        v7 = self.dropout2(self.fcbus2(spk7_1))
        spk7_2, mem7_2 = self.lifbus(v7, mem7_2)

        v8 = self.dropout2(self.fctruck1(spk3_flat))
        spk8_1, mem8_1 = self.lif7(v8, mem8_1)
        v8 = self.dropout2(self.fctruck2(spk8_1))
        spk8_2, mem8_2 = self.liftruck(v8, mem8_2)

        return (
            spk5_2,
            spk6_2,
            spk7_2,
            spk8_2,
            (
                mem1,
                mem2,
                mem3,
                mem5_1,
                mem5_2,
                mem6_1,
                mem6_2,
                mem7_1,
                mem7_2,
                mem8_1,
                mem8_2,
            ),
        )


def loss_fn(output_frame, target_frame, step, class_id):
    mse_loss = 0
    mse_loss += loss_function(
        output_frame, target_frame * 1000
    )  # Multiplication due to the numbers being too small, should be fixed when creating the data

    # if print_image:
    #     save_current_result(output_frame, target_frame, frames, step, class_id)

    return mse_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SNN()


model.load_state_dict(
    torch.load("models/new-model-35.45.pth", weights_only=True, map_location="cuda:0")
)
model = model.to(device)
model.eval()


data = torch.load(f"training_data/event_frames_16.pt")
# f"./w31/box2/2-07-31/events/event_frames_14.pt"
# f"./w31/box2/2-08-01/events/event_frames_4.pt"
# f"./w38/box3/3-09-27/events/event_frames_6.pt"


mem_states = (None, None, None, None, None, None, None, None, None, None, None)

fig, ax = plt.subplots(1, 5, figsize=(30, 10))  # , layout="compressed")


ax[0].axis("off")
ax[1].axis("off")
ax[2].axis("off")
ax[3].axis("off")
ax[4].axis("off")


event_img = ax[0].imshow([[0]], cmap="gray")

output_img1 = ax[1].imshow(
    [[0]],
    cmap="magma",
    vmin=0,
    vmax=1,
)
output_img2 = ax[2].imshow(
    [[0]],
    cmap="magma",
    vmin=0,
    vmax=1,
)
output_img3 = ax[3].imshow(
    [[0]],
    cmap="magma",
    vmin=0,
    vmax=1,
)
output_img4 = ax[4].imshow(
    [[0]],
    cmap="magma",
    vmin=0,
    vmax=1,
)


# cbar = fig.colorbar(
#     output_img4,
# )


FRAMES = len(data)


def animate(step):
    global mem_states
    if step == FRAMES - 1:
        plt.close(fig)
    # gc.collect()
    # torch.cuda.empty_cache()
    with torch.no_grad():
        frame = torch.tensor(
            np.array(np.array([[data[step].to_dense()[160 + 28 : 160 + 256 - 28, 20 : 20 + 200]]])),
            device=device,
        )  # [180:480, 100:400]
        frame.to(device)

        with torch.amp.autocast(device_type="cuda"):
            output1, output2, output3, output4, mem_states = model(frame, mem_states)
        # print(output.shape)
        final_output1 = output1.view(1, 64, 64)
        final_output2 = output2.view(1, 64, 64)
        final_output3 = output3.view(1, 64, 64)
        final_output4 = output4.view(1, 64, 64)

        event_img.set_data(frame[0][0].cpu().detach().numpy())
        event_img.autoscale()

        maximum = np.max(
            [
                torch.max(final_output1[0]).item(),
                torch.max(final_output2[0]).item(),
                torch.max(final_output3[0]).item(),
                torch.max(final_output4[0]).item(),
            ]
        )
        minimum = 2

        output_img1.set(
            data=final_output1[0].cpu().detach().numpy(),
            # clim=(minimum, max(minimum, maximum)),
            clim=(1, max(1, torch.max(final_output1[0]).item())),
        )

        output_img2.set(
            data=final_output2[0].cpu().detach().numpy(),
            # clim=(minimum, max(minimum, maximum)),
            clim=(minimum, max(minimum, torch.max(final_output2[0]).item())),
        )

        output_img3.set(
            data=final_output3[0].cpu().detach().numpy(),
            # clim=(minimum, max(minimum, maximum)),
            clim=(minimum, max(minimum, torch.max(final_output3[0]).item())),
        )

        output_img4.set(
            data=final_output4[0].cpu().detach().numpy(),
            # clim=(minimum, max(minimum, maximum)),
            clim=(minimum, max(minimum, torch.max(final_output4[0]).item())),
        )

    # frame = None
    # output = None
    # final_output = None
    # del frame, output, final_output
    return (
        event_img,
        output_img1,
        output_img2,
        output_img3,
        output_img4,
    )
    # fig.colorbar(output, ax=ax[1], location="right")


ani = animation.FuncAnimation(
    fig,
    partial(animate),
    frames=FRAMES,
    interval=1,
    repeat=False,
    blit=True,
)

# plt.show()

save_dir = "test-anim-multi-full-dataset-maybework-deepnet10.mp4"  # os.path.join(os.path.dirname(sys.argv[0]), 'MyVideo.mp4')
print(f"Saving video to {save_dir}...")
video_writer = animation.FFMpegWriter(fps=90, bitrate=-1)
update_func = lambda _i, _n: progress_bar.update(1)
with tqdm(total=len(data), ncols=86, desc="Saving video") as progress_bar:
    ani.save(save_dir, writer=video_writer, dpi=100, progress_callback=update_func)
print("Video saved successfully.")
