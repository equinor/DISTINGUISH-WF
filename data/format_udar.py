from pathlib import Path
from tkinter import Tk, filedialog, simpledialog

import pandas as pd

# Hide Tk root window
root = Tk()
root.withdraw()

# Select file
input_file = filedialog.askopenfilename(
    title="Select UDAR file",
    filetypes=[("Text files", "*.txt")]
)

if not input_file:
    raise SystemExit("No file selected.")

# Ask MD range
md_from = simpledialog.askfloat("MD Start", "Enter FROM MD:")
md_to = simpledialog.askfloat("MD End", "Enter TO MD:")

if md_from is None or md_to is None:
    raise SystemExit("MD interval not specified.")

# Read GeoSphere header
with open(input_file, "r") as f:
    header = f.readline().strip()

header = header.lstrip("%").strip()
columns = header.split()

# Read data
df = pd.read_csv(
    input_file,
    sep=r"\s+",
    skiprows=1,
    names=columns
)

# Clip interval
df_clip = df[
    (df["MD"] >= md_from) &
    (df["MD"] <= md_to)
]

# Output filename
input_path = Path(input_file)

output_file = (
    input_path.parent
    / f"{input_path.stem}_{int(md_from)}to{int(md_to)}.txt"
)

# Write file preserving format
with open(output_file, "w") as f:
    f.write("% " + " ".join(columns) + "\n")

    df_clip.to_csv(
        f,
        sep=" ",
        index=False,
        header=False,
        float_format="%.6f"
    )

print(f"Saved: {output_file}")
print(f"Rows exported: {len(df_clip)}")
