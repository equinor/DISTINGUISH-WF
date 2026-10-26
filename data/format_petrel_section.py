import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import torch

PETREL_FILE = r"C:\Appl\JDGO\real_data\RM1_MC3_1R456_2R123_H_rees"

TRAJ_FILE = r"C:\Appl\JDGO\real_data\B18T2_survey_Zmap"


def load_petrel(filename):

    points = []

    after_header = False

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if line == "END HEADER":
                after_header = True
                continue

            if not after_header:
                continue

            if not line:
                continue

            vals = line.split()

            if len(vals) < 4:
                continue

            points.append([
                float(vals[0]),
                float(vals[1]),
                float(vals[2]),
                int(float(vals[3]))
            ])

    return pd.DataFrame(
        points,
        columns=["X", "Y", "Z", "FACIES"]
    )


def load_trajectory(filename):

    points = []

    with open(filename, "r") as f:

        for line in f:

            line = line.strip()

            if (
                not line
                or line.startswith("#")
                or line.startswith("B18T2")
            ):
                continue

            vals = line.split()

            if len(vals) < 4:
                continue

            points.append([
                float(vals[0]),   # X
                float(vals[1]),   # Y
                float(vals[3])    # Z
            ])

    return pd.DataFrame(
        points,
        columns=["X", "Y", "Z"]
    )

def compute_section_length(df):

    unique_xy = (
        df[["X", "Y"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    dx = np.sqrt(
        np.diff(unique_xy["X"])**2 +
        np.diff(unique_xy["Y"])**2
    )

    print(
        "Mean horizontal spacing:",
        dx.mean()
    )

    z_levels = np.sort(df["Z"].unique())

    dz = np.abs(np.diff(z_levels))

    print(
        "Mean vertical spacing:",
        dz.mean()
    )

    md = [0.0]

    for i in range(1, len(unique_xy)):

        dx = (
            unique_xy.loc[i, "X"]
            - unique_xy.loc[i - 1, "X"]
        )

        dy = (
            unique_xy.loc[i, "Y"]
            - unique_xy.loc[i - 1, "Y"]
        )

        md.append(
            md[-1] +
            np.sqrt(dx**2 + dy**2)
        )

    unique_xy["SECTION_MD"] = md

    return unique_xy


def build_section(df):

    print("\nBuilding section...")

    unique_xy = compute_section_length(df)

    print(
        f"Number of XY columns: "
        f"{len(unique_xy)}"
    )

    print(
        f"Section length: "
        f"{unique_xy['SECTION_MD'].max():.1f} m"
    )

    column_lookup = {
        (row.X, row.Y): idx
        for idx, row in unique_xy.iterrows()
    }

    df["COLUMN"] = [
        column_lookup[(x, y)]
        for x, y in zip(df["X"], df["Y"])
    ]

    pivot = df.pivot_table(
        index="Z",
        columns="COLUMN",
        values="FACIES",
        aggfunc="first"
    )

    pivot = pivot.sort_index(
        ascending=False
    )

    print(
        f"Raw pivot shape: "
        f"{pivot.shape}"
    )

    pivot = pivot.ffill().bfill()

    facies = pivot.to_numpy()

    unique_xy["Z_TOP"] = pivot.index.max()
    unique_xy["Z_BOTTOM"] = pivot.index.min()

    return facies, unique_xy


def majority_vote_coarsen(
    facies,
    x_factor=10,
    z_factor=46
):

    nz, nx = facies.shape

    nz_out = nz // z_factor
    nx_out = nx // x_factor

    coarse = np.zeros(
        (nz_out, nx_out),
        dtype=np.int32
    )

    for iz in range(nz_out):

        z0 = iz * z_factor
        z1 = z0 + z_factor

        for ix in range(nx_out):

            x0 = ix * x_factor
            x1 = x0 + x_factor

            block = facies[
                z0:z1,
                x0:x1
            ]

            values = block.flatten()

            values = values[
                ~np.isnan(values)
            ]

            if len(values) == 0:
                coarse[iz, ix] = 0
                continue

            coarse[iz, ix] = (
                Counter(values)
                .most_common(1)[0][0]
            )

    return coarse


def map_trajectory_to_section(
    traj_df,
    section_info,
    facies
):

    traj_cols = []
    traj_rows = []

    z_top = section_info["Z_TOP"].iloc[0]
    z_bottom = section_info["Z_BOTTOM"].iloc[0]

    nz = facies.shape[0]

    for _, row in traj_df.iterrows():

        dist2 = (
            (section_info["X"] - row["X"])**2
            +
            (section_info["Y"] - row["Y"])**2
        )

        col = dist2.idxmin()

        traj_cols.append(col)

        # trajectory Z positive
        # section Z negative

        z_section = -row["Z"] + 1.5

        row_idx = (
            (z_section - z_top)
            /
            (z_bottom - z_top)
        ) * (nz - 1)

        traj_rows.append(row_idx)

    return (
        np.array(traj_cols),
        np.array(traj_rows)
    )

def show_section(
    facies,
    title,
    traj_cols=None,
    traj_rows=None
):

    plt.figure(figsize=(14, 6))

    plt.imshow(
        facies,
        aspect="auto",
        cmap="tab10",
        interpolation="nearest",
        origin="upper"
    )

    if (
        traj_cols is not None
        and traj_rows is not None
    ):

        plt.plot(
            traj_cols,
            traj_rows,
            "w-",
            linewidth=2
        )

    plt.colorbar(label="Facies")

    plt.title(title)

    plt.xlabel("Column")
    plt.ylabel("Depth")

    plt.tight_layout()

    plt.show()


def extract_distinguish_window(
    facies_coarse,
    traj_cols_coarse=None,
    traj_rows_coarse=None,
    center_col=78,
    center_row=87
):

    total_rows, total_cols = facies_coarse.shape

    col0 = center_col - 32
    row0 = center_row - 32

    col0 = max(
        0,
        min(col0, total_cols - 64)
    )

    row0 = max(
        0,
        min(row0, total_rows - 64)
    )

    row1 = row0 + 64
    col1 = col0 + 64

    print(f"Window rows {row0}:{row1}")
    print(f"Window cols {col0}:{col1}")

    window = facies_coarse[
        row0:row1,
        col0:col1
    ]

    # crop trajectory too
    traj_cols_window = None
    traj_rows_window = None

    if (
        traj_cols_coarse is not None
        and traj_rows_coarse is not None
    ):

        mask = (
            (traj_cols_coarse >= col0)
            &
            (traj_cols_coarse < col1)
            &
            (traj_rows_coarse >= row0)
            &
            (traj_rows_coarse < row1)
        )

        traj_cols_window = (
            traj_cols_coarse[mask] - col0
        )

        traj_rows_window = (
            traj_rows_coarse[mask] - row0
        )

    return (
        window,
        traj_cols_window,
        traj_rows_window
    )


def main():

    print("Loading Petrel file...")

    df = load_petrel(PETREL_FILE)

    print(df.head())

    traj_df = load_trajectory(
        TRAJ_FILE
    )

    print(
        "\nTrajectory stations:",
        len(traj_df)
    )

    facies, section_info = build_section(df)

    facies_coarse = majority_vote_coarsen(
        facies,
        x_factor=10,
        z_factor=46
    )

    traj_cols, traj_rows = (
        map_trajectory_to_section(
            traj_df,
            section_info,
            facies
        )
    )

    print(
        "Trajectory column range:",
        traj_cols.min(),
        traj_cols.max()
    )

    print(
        "Trajectory row range:",
        traj_rows.min(),
        traj_rows.max()
    )

    traj_cols_coarse = (
        traj_cols / 10.0
    )

    traj_rows_coarse = (
        traj_rows / 46.0
    )

    show_section(
        facies_coarse,
        "Coarsened Section + Trajectory",
        traj_cols=traj_cols_coarse,
        traj_rows=traj_rows_coarse
    )

    facies64, traj_cols64, traj_rows64 = (
        extract_distinguish_window(
            facies_coarse,
            traj_cols_coarse,
            traj_rows_coarse,
            center_col=78,
            center_row=87
        )
    )

    print(
        "Distinguish window shape:",
        facies64.shape
    )

    show_section(
        facies64,
        "Distinguish Window (64x64)",
        traj_cols=traj_cols64,
        traj_rows=traj_rows64
    )

    #
    # Save facies window
    #
    np.save(
        "petrel_facies_64.npy",
        facies64
    )

    #
    # Build GAN-equivalent tensor
    # Shape = [1,6,64,64]
    #
    images = np.zeros(
        (1, 6, 64, 64),
        dtype=np.float32
    )

    images[0, 0] = (
        facies64 == 0
    ).astype(np.float32)

    images[0, 1] = (
        facies64 == 1
    ).astype(np.float32)

    images[0, 2] = (
        facies64 == 2
    ).astype(np.float32)

    # channels 3-5 unused by proxy
    images[0, 3] = 0.0
    images[0, 4] = 0.0
    images[0, 5] = 0.0

    images = torch.tensor(
        images,
        dtype=torch.float32
    )

    print(
        "Proxy tensor shape:",
        images.shape
    )

    torch.save(
        images,
        "petrel_proxy_tensor.pt"
    )

    print(
        "\nSaved:"
    )

    print(
        "  petrel_facies_64.npy"
    )

    print(
        "  petrel_proxy_tensor.pt"
    )


if __name__ == "__main__":
    main()
