# Raw Passive-Foot Bench Curves

These are the five byte-preserved GB18030 CSV exports used as the passive-foot fit inputs. They are the raw evidence; `fit_params.txt`, `fit_rmse.csv`, URDF, and MJCF files are derived from later processing.

| File | Source label | Position | Columns |
| --- | --- | ---: | --- |
| `a3_toe_cantilever_compression_A_X90mm.csv` | A | 90 mm | time (s), displacement (mm), force (kN), compressive strain (%) |
| `a3_toe_cantilever_compression_B_X117mm.csv` | B | 117 mm | same |
| `a3_toe_cantilever_compression_C_X140mm.csv` | C | 140 mm | same |
| `a3_toe_cantilever_compression_D_X160mm.csv` | D | 160 mm | same |
| `a3_toe_cantilever_compression_E_X182mm.csv` | E | 182 mm | same |

Read with `encoding="gb18030"`. `raw_manifest.csv` records the source archive member, byte size, and SHA-256 for each file. The source labels are not a confirmed common fixture coordinate system; do not reinterpret them. These measurements are quasi-static and do not establish dynamic damping, complete fixture calibration, or a causal sim2real effect.
