# DISTINGUISH RealBridge - Context (up to `main-realbridge` branch) (20261002 13h00)

## Project Stage

Current phase:

> **Real Data Integration & Validation**

This is no longer setup/debugging. The synthetic DISTINGUISH workflow is functioning and we're building a bridge that allows real GeoSphere HD UDAR measurements to drive the assimilation workflow.

**Branch:**

```text
main-realbridge
```

---

# Objective

Replace:

```text
Synthetic latent truth
    ↓
GeoSim synthetic observation generation
    ↓
data.pkl / var.pkl
```

with:

```text
Real GeoSphere HD UDAR measurements
    ↓
RealTruth
    ↓
data.pkl / var.pkl
```

while keeping unchanged:

```text
GAN
GeoSim forward simulator
EM proxy
PET/PIPT
Dynamic Programming
Path Optimizer
Robot steering logic
```

## Core philosophy

> Adapt the real data to the existing DISTINGUISH observation contract instead of redesigning the workflow.

---

# Critical Discovery

Initially assumed:

```text
MD → (row,column)
```

This was incorrect.

The actual problem is:

```text
MD
 ↓
Assimilation position
 ↓
48-channel observation vector
```

We are dealing with signals, not geological images.

The observation space is simply:

```text
48 UDAR measurements
```

evaluated at successive drilling positions.

The row dimension is **not**:

```text
facies
resistivity
depth
TVD
```

---

# GeoSphere HD Knowledge

## Frequency mapping

```text
F1 = 2 kHz
F2 = 6 kHz
F3 = 12 kHz
F4 = 24 kHz
F5 = 48 kHz
F6 = 96 kHz
```

## Receiver spacing

```text
R1 = 43 ft
R2 = 83 ft
```

## Example

```text
USDA_F2_R2
```

means:

```text
USDA
6 kHz
83 ft receiver
```

---

# DISTINGUISH Observation Contract

The workflow expects six configurations:

```python
[
 ('6kHz',  '83ft'),
 ('12kHz', '83ft'),
 ('24kHz', '83ft'),
 ('24kHz', '43ft'),
 ('48kHz', '43ft'),
 ('96kHz', '43ft')
]
```

For each configuration it expects:

```python
[
 USDA,
 USDP,
 UADA,
 UADP,
 UHRA,
 UHRP,
 UHAA,
 UHAP
]
```

Therefore:

```text
6 configurations × 8 signals = 48 observations
```

per assimilation step.

---

# Verified Real ↔ Synthetic Mapping

```text
6 kHz / 83 ft   -> F2_R2
12 kHz / 83 ft  -> F3_R2
24 kHz / 83 ft  -> F4_R2
24 kHz / 43 ft  -> F4_R1
48 kHz / 43 ft  -> F5_R1
96 kHz / 43 ft  -> F6_R1
```

Signal order must remain exactly:

```text
USDA
USDP
UADA
UADP
UHRA
UHRP
UHAA
UHAP
```

---

# Verification Performed

File:

```text
Awell_240426_LS_channelresponse.txt
```

Verification result:

```text
All 48 required channels are present.
```

No channel mapping blockers remain.

---

# SyntheticTruth Reverse Engineering

Inspection of:

```python
SyntheticTruth.acquire_data()
```

revealed:

```python
logs_np.shape == (6, 8)
```

Structure:

```python
logs_np[config_idx, signal_idx]
```

contains:

```text
USDA
USDP
UADA
UADP
UHRA
UHRP
UHAA
UHAP
```

The remainder of the method serializes this structure into:

```text
data.pkl
var.pkl
var.csv
assim_index.csv
datatyp.csv
```

## Key conclusion

RealTruth only needs to generate:

```python
logs_np.shape == (6,8)
```

and everything downstream can remain unchanged.

---

# RealTruth Design

RealTruth is intended as a drop-in replacement for:

```python
SyntheticTruth
```

## Interface

```python
RealTruth.acquire_data(keys)
```

## Responsibilities

```text
1. Read GeoSphere file
2. Determine drilling position
3. Extract 48 channels
4. Assemble logs_np (6,8)
5. Write data.pkl
6. Write var.pkl
```

---

# MD Mapping Strategy

Real data:

```text
1 m sampling
```

Synthetic workflow:

```text
10 m sampling
```

Current mapping:

```python
real_idx = bit_pos_column * 10
```

Examples:

```text
column 0 -> row 0
column 1 -> row 10
column 2 -> row 20
```

This is the current implementation assumption.

---

# Streamlit Changes Implemented

Added a runtime mode selector.

## UI

```text
☐ Use Real UDAR Data
```

When enabled:

```text
Upload GeoSphere HD file
```

## Behaviour

```text
Synthetic mode
    ↓
SyntheticTruth

Real mode
    ↓
File upload
    ↓
RealTruth
```

## Benefits

```text
No hardcoded paths
No synthetic assumptions
Easy A/B testing
```

---

# Cheat Mode

Original code depends on:

```python
true_sim.simulator
true_sim.latent_synthetic_truth
```

which do not exist in RealTruth.

## Fix

```python
if (
    not st.session_state.is_real_data
    and st.checkbox("Cheat!")
):
```

Result:

```text
Cheat mode available only in synthetic mode.
```

---

# Header Parsing Issue (Solved)

Source file begins with:

```text
% MD TVD THL USDA_F1_R1 ...
```

Original parser:

```python
pd.read_csv(
    ...,
    comment='%'
)
```

## Problem

Entire header line was discarded.

Observed symptom:

```python
df.columns
```

became:

```text
4200.000000
2096.258151
3134.653621
...
```

because the first data row was interpreted as the header.

## Solution

```python
with open(udar_file, "r") as f:
    header = f.readline().strip()

if header.startswith("%"):
    header = header[1:].strip()

columns = header.split()

self.df = pd.read_csv(
    udar_file,
    sep=r"\s+",
    skiprows=1,
    names=columns
)
```

Result:

```text
Correct GeoSphere channel names loaded.
```

---

# Caching Decision

Keep:

```python
@st.cache_data
def get_gan_earth(...)
```

Disable:

```python
@st.cache_data
def da(...)
```

## Reason

`da()` has side effects:

```text
writes data.pkl
writes var.pkl
runs PET/PIPT
reads posterior results
```

and is therefore not a safe caching candidate during RealTruth development.

---

# Utility Created

## UDAR Formatting / Clipping Utility

Recommended name:

```text
format_udar.py
```

Purpose:

```text
Select file
Enter MD interval
Export clipped file
```

Output example:

```text
Awell_240426_LS_channelresponse_4200to4600.txt
```

stored beside the original file.

---

# Current RealBridge Status

## Completed

- ✅ Real-data branch established
- ✅ GeoSphere frequency mapping
- ✅ Receiver spacing mapping
- ✅ Verified 48-channel mapping
- ✅ SyntheticTruth reverse engineering
- ✅ RealTruth architecture defined
- ✅ Streamlit real-data mode
- ✅ File upload workflow
- ✅ Cheat isolation
- ✅ Header parsing diagnosis
- ✅ Header parsing fix
- ✅ MD downsampling strategy
- ✅ UDAR clipping utility

---

# Remaining Development Work

## 1. Validate RealTruth Extraction

Confirm:

```python
logs_np.shape == (6,8)
```

Inspect actual values.

Recommended debug:

```python
print(logs_np.shape)
print(logs_np)
```

---

## 2. Validate data.pkl

Compare:

```text
SyntheticTruth -> data.pkl
```

versus:

```text
RealTruth -> data.pkl
```

Structure must match.

---

## 3. Validate var.pkl

Compare:

```text
SyntheticTruth -> var.pkl
```

versus:

```text
RealTruth -> var.pkl
```

Structure must match.

Current temporary variance model:

```python
(0.1 * max(abs(values)))**2
```

per signal.

---

## 4. First Assimilation Test

Goal:

```text
Upload real UDAR
    ↓
Acquire observation
    ↓
PET/PIPT runs
    ↓
Posterior generated
```

No concern yet about geological correctness.

Focus only on execution.

---

## 5. Multi-Step Assimilation Test

Test:

```text
Drill
Assimilate**rill
Assimilate
...
```

Verify:
** MD indexing remains valid
- Corr**t measurements are selected
- No **t-of-range access occurs

---

# **iding Principle

Do **not** modif**

```text
GeoSim
GAN
EM proxy
PET**IPT
DP planner
Path optimizer
```**Focus exclusively on:

```text
Ge**phere HD file
    ↓
RealTruth**   ↓
data.pkl / var.pkl
```

**e RealBridge succeeds if RealTrut**can produce the exact observation**ontract already expected by the existing DISTINGUISH workflow.