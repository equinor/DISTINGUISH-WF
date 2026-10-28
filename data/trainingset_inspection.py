import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import (
    Slider,
    Button,
    TextBox
)
from matplotlib.colors import ListedColormap


def windows(data):

    print("Dataset shape:", data.shape)

    cmap = ListedColormap([
        "dimgray",      # 0 = shale
        "gold",         # 1 = sand
        "deepskyblue"   # 2 = sandy shale
    ])

    current_idx = 0

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.subplots_adjust(
        bottom=0.22
    )

    img = ax.imshow(
        data[current_idx],
        cmap=cmap,
        vmin=0,
        vmax=2,
        interpolation="nearest",
        aspect=0.2
    )

    ax.axis("off")

    title = ax.set_title("")

    def refresh(idx):

        img.set_data(data[idx])

        shale = np.sum(data[idx] == 0)
        sand  = np.sum(data[idx] == 1)
        splay = np.sum(data[idx] == 2)

        title.set_text(
            f"Window {idx+1}/{len(data)} | "
            f"Shale={shale} "
            f"Sand={sand} "
            f"Splay={splay}"
        )

        fig.canvas.draw_idle()

    # --------------------------------------------------
    # Slider
    # --------------------------------------------------

    slider_ax = plt.axes(
        [0.15, 0.12, 0.70, 0.03]
    )

    slider = Slider(
        slider_ax,
        "Window",
        0,
        len(data)-1,
        valinit=0,
        valstep=1
    )

    # --------------------------------------------------
    # Previous button
    # --------------------------------------------------

    prev_ax = plt.axes(
        [0.15, 0.04, 0.10, 0.05]
    )

    prev_btn = Button(
        prev_ax,
        "◀ Prev"
    )

    # --------------------------------------------------
    # Next button
    # --------------------------------------------------

    next_ax = plt.axes(
        [0.27, 0.04, 0.10, 0.05]
    )

    next_btn = Button(
        next_ax,
        "Next ▶"
    )

    # --------------------------------------------------
    # Jump textbox
    # --------------------------------------------------

    box_ax = plt.axes(
        [0.55, 0.04, 0.20, 0.05]
    )

    textbox = TextBox(
        box_ax,
        "Go to:"
    )

    # --------------------------------------------------
    # Callbacks
    # --------------------------------------------------

    def slider_update(val):

        idx = int(slider.val)
        refresh(idx)

    slider.on_changed(slider_update)

    def prev(event):

        idx = max(
            0,
            int(slider.val) - 1
        )

        slider.set_val(idx)

    prev_btn.on_clicked(prev)

    def next_(event):

        idx = min(
            len(data)-1,
            int(slider.val) + 1
        )

        slider.set_val(idx)

    next_btn.on_clicked(next_)

    def goto(text):

        try:

            idx = int(text) - 1

            if 0 <= idx < len(data):

                slider.set_val(idx)

        except ValueError:
            pass

    textbox.on_submit(goto)

    # --------------------------------------------------
    # Keyboard support
    # --------------------------------------------------

    def key_press(event):

        idx = int(slider.val)

        if event.key == "right":

            slider.set_val(
                min(
                    len(data)-1,
                    idx + 1
                )
            )

        elif event.key == "left":

            slider.set_val(
                max(
                    0,
                    idx - 1
                )
            )

    fig.canvas.mpl_connect(
        "key_press_event",
        key_press
    )

    # --------------------------------------------------
    # Legend
    # --------------------------------------------------

    handles = [
        plt.Rectangle(
            (0,0),1,1,
            color="dimgray"
        ),
        plt.Rectangle(
            (0,0),1,1,
            color="gold"
        ),
        plt.Rectangle(
            (0,0),1,1,
            color="deepskyblue"
        )
    ]

    ax.legend(
        handles,
        [
            "Shale",
            "Sand",
            "Sandy Shale"
        ],
        loc="upper right"
    )

    refresh(0)

    plt.show()

def verify_Values(data):
    print(data.shape)
    print(data.dtype)

    print("Min:", data.min())
    print("Max:", data.max())

    print("Unique values:")
    print(np.unique(data))

def frac_data(data):
    frac0 = (
        data == 0
    ).mean(axis=(1,2))

    print(
        "Windows >90% facies 0:",
        (frac0 > 0.9).sum()
    )

    print(
        "Windows >95% facies 0:",
        (frac0 > 0.95).sum()
    )

    print(
        "Windows >99% facies 0:",
        (frac0 > 0.99).sum()
    )

def check_data(data):
    print(data.shape)
    print(data.dtype)
    print(np.unique(data))

    vals, counts = np.unique(data, return_counts=True)
    for v, c in zip(vals, counts):
        print(v, c, c/counts.sum())

data = np.load("training_dataset/training_windows_T1.npy")
#check_data(data)
windows(data)

