from pathlib import Path
import re
import numpy as np
from scipy.ndimage import map_coordinates

# =============================================================================
# USER SETTINGS
# =============================================================================

CONFIG = {
    "grdecl_file": r"C:\Appl\JDGO\DISTINGUISH-WF-main\data\GK_training_grid_10x10x05\GKjdgo.GRDECL",

    "azimuth_step": 10,

    "window_size": 64,
    "window_stride_xy": 32,
    "window_stride_z": 16,

    "section_spacing": 2,

    "padding": 32,

    "max_inactive_fraction": 0.50,
    "max_facies0_fraction": 0.95,
    "max_facies1_fraction": 0.80,

    "output_dir": "training_dataset"
}

# =============================================================================
# GRDECL PARSER
# =============================================================================


class GRDECLParser:

    def __init__(self, filepath):

        self.filepath = Path(filepath)

        self.data = {}

        self.nx = None
        self.ny = None
        self.nz = None

    def load(self):

        text = self._read_with_includes(
            self.filepath
        )

        self._parse_text(text)

        self._read_specgrid()

        return self

    def _read_with_includes(self, filepath):

        result = []

        with open(filepath, "r") as f:

            lines = f.readlines()

        directory = filepath.parent

        i = 0

        while i < len(lines):

            line = lines[i]

            clean = line.split("--")[0].strip()

            if clean.upper().startswith("INCLUDE"):

                include_text = ""

                j = i + 1

                while j < len(lines):

                    include_text += lines[j]

                    if "/" in lines[j]:
                        break

                    j += 1

                m = re.search(
                    r"'([^']+)'",
                    include_text
                )

                if m:

                    include_file = (
                        directory /
                        m.group(1)
                    )

                    print(
                        f"Loading include:"
                        f" {include_file.name}"
                    )

                    result.append(
                        self._read_with_includes(
                            include_file
                        )
                    )

                i = j + 1
                continue

            result.append(line)

            i += 1

        return "\n".join(result)

    def _parse_text(self, text):

        lines = text.splitlines()

        current_keyword = None
        values = []

        for raw in lines:

            line = raw.split("--")[0].strip()

            if not line:
                continue

            if current_keyword is None:

                token = line.split()[0]

                if re.match(
                    r"^[A-Z][A-Z0-9_]*$",
                    token
                ):

                    current_keyword = token

                    rest = line[len(token):].strip()

                    if rest:

                        values.extend(
                            rest.split()
                        )

                    if "/" in values:

                        idx = values.index("/")

                        self.data[
                            current_keyword
                        ] = self._expand(
                            values[:idx]
                        )

                        current_keyword = None
                        values = []

                    continue

            else:

                values.extend(
                    line.split()
                )

                if "/" in values:

                    idx = values.index("/")

                    self.data[
                        current_keyword
                    ] = self._expand(
                        values[:idx]
                    )

                    current_keyword = None
                    values = []

    def _expand(self, tokens):

        expanded = []

        for token in tokens:

            if "*" in token:

                n, value = token.split("*")

                n = int(n)

                value = self._cast(value)

                expanded.extend(
                    [value] * n
                )

            else:

                expanded.append(
                    self._cast(token)
                )

        return np.asarray(expanded)

    def _cast(self, value):

        try:

            if (
                "." in value
                or "e" in value.lower()
            ):
                return float(value)

            return int(value)

        except:

            return value

    def _read_specgrid(self):

        if "SPECGRID" not in self.data:
            raise RuntimeError(
                "SPECGRID not found"
            )

        spec = self.data["SPECGRID"]

        self.nx = int(spec[0])
        self.ny = int(spec[1])
        self.nz = int(spec[2])

        print(
            f"NX={self.nx} "
            f"NY={self.ny} "
            f"NZ={self.nz}"
        )

    def get_property(self, keyword):

        keyword = keyword.upper()

        if keyword not in self.data:

            raise KeyError(
                f"{keyword} not found"
            )

        arr = self.data[keyword]

        expected = (
            self.nx *
            self.ny *
            self.nz
        )

        if len(arr) != expected:

            raise ValueError(
                f"{keyword}: "
                f"{len(arr):,} values "
                f"found, expected "
                f"{expected:,}"
            )

        return arr.reshape(
            (
                self.nx,
                self.ny,
                self.nz
            ),
            order="F"
        )


# =============================================================================
# SECTION EXTRACTION
# =============================================================================


def extract_section(
    cube,
    x0,
    y0,
    azimuth_deg,
    padding
):

    nx, ny, nz = cube.shape

    az = np.radians(
        azimuth_deg
    )

    length = int(
        np.hypot(nx, ny)
    )

    s = np.arange(
        0,
        length
        )

    x = (
        x0 +
        s * np.cos(az)
    )

    y = (
        y0 +
        s * np.sin(az)
    )

    z = np.arange(nz)

    xx, zz = np.meshgrid(
        x,
        z,
        indexing="ij"
    )

    yy, _ = np.meshgrid(
        y,
        z,
        indexing="ij"
    )

    coords = np.vstack([
        xx.ravel(),
        yy.ravel(),
        zz.ravel()
    ])

    values = map_coordinates(
        cube,
        coords,
        order=0,
        mode="constant",
        cval=0
    )

    section = values.reshape(
        len(s),
        nz
    )

    section = np.pad(
        section,
        (
            (0, 0),
            (padding, padding) # (padding, padding) # To pad up and down, use (padding, padding)
        ),
        mode="constant",
        constant_values=0
    )


    return section


# =============================================================================
# MAIN
# =============================================================================


def main():

    out_dir = Path(
        CONFIG["output_dir"]
    )

    out_dir.mkdir(
        exist_ok=True
    )

    parser = (
        GRDECLParser(
            CONFIG["grdecl_file"]
        )
        .load()
    )

    facies = parser.get_property(
        "FACIES"
    )

    actnum = parser.get_property(
        "ACTNUM"
    )

    print(
        "\nFACIES shape:",
        facies.shape
    )

    print(
        "ACTNUM shape:",
        actnum.shape
    )

    samples = []

    nx, ny, nz = facies.shape

    for azimuth in range(
        0,
        180,
        CONFIG["azimuth_step"]
    ):

        print(f"\nAzimuth {azimuth}")

        az_rad = np.radians(azimuth)

        nx_norm = -np.sin(az_rad)
        ny_norm =  np.cos(az_rad)

        cx = nx / 2.0
        cy = ny / 2.0

        max_offset = min(nx, ny) // 2

        for offset in range(
            -max_offset,
            max_offset + 1,
            CONFIG["section_spacing"]
        ):

            x0 = cx + offset * nx_norm
            y0 = cy + offset * ny_norm

            fac_sec = extract_section(
                facies,
                x0,
                y0,
                azimuth,
                CONFIG["padding"]
            )

            act_sec = extract_section(
                actnum,
                x0,
                y0,
                azimuth,
                CONFIG["padding"]
            )

            sx, sz = fac_sec.shape

            for ix in range(
                0,
                sx - CONFIG["window_size"] + 1,
                CONFIG["window_stride_xy"]
            ):
                for iz in range(
                    0,
                    sz - CONFIG["window_size"] + 1,
                    CONFIG["window_stride_z"]
                ):

                    fwin = fac_sec[
                        ix:ix + CONFIG["window_size"],
                        iz:iz + CONFIG["window_size"]
                    ]

                    awin = act_sec[
                        ix:ix + CONFIG["window_size"],
                        iz:iz + CONFIG["window_size"]
                    ]

                    inactive_fraction = (awin == 0).mean()
                    facies0_fraction = (fwin == 0).mean()
                    facies1_fraction = (fwin == 1).mean()

                    if inactive_fraction > CONFIG["max_inactive_fraction"]:
                        continue

                    if facies0_fraction > CONFIG["max_facies0_fraction"]:
                        continue
                    if facies1_fraction > CONFIG["max_facies1_fraction"]:
                                            continue

                    # Mirror of the window based on sample to unbias the dipping direction
                    if len(samples) % 2 == 1:
                        fwin = np.fliplr(fwin)
                    # Save samples
                    samples.append(
                        fwin.astype(np.int16)
                    )
                    # Mirror of the window to unbias the diping direction
                    #samples.append( 
                    #    np.fliplr(fwin).astype(np.int16)
                    #)


    samples = np.asarray(
        samples,
        dtype=np.int16
    )

    print(
        "\nTotal windows:",
        len(samples)
    )

    output_file = (
        out_dir /
        "training_windows.npy"
    )

    np.save(
        output_file,
        samples
    )

    print(
        f"\nSaved: "
        f"{output_file}"
    )

    print(
        "Dataset shape:",
        samples.shape
    )


if __name__ == "__main__":

    main()