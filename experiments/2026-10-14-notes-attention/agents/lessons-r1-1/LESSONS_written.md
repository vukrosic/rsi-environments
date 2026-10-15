The original kernel was reported as 67.6 milliseconds. All ten benchmark runs were used. Reference times varied from 57.91 to 72.17 milliseconds, so compare absolute kernel times as well as speedups.

Built-in causal scaled dot product attention took 17.01 milliseconds in float32 and 10.64 milliseconds with half inputs and a float output. Both passed.

The fastest tested implementation was a custom Metal tiled attention shader using SIMD group matrix operations. Each SIMD group handles eight query rows and all 64 output columns. Stage queries, keys, and values as half precision; accumulate scores and outputs in float32. Keep online softmax maxima and denominators in float32, reduce each row using lane shuffles, and skip wholly future key tiles. This avoids materializing the full attention matrix. Compile the shader once at import.

Query/key tile sizes and times: 32 by 64, 8.04 milliseconds; 64 by 64, 7.04 milliseconds; 64 by 32, 7.53 milliseconds; 128 by 64, 10.61 milliseconds. The final kernel restores 64 by 64, which achieved an 8.23-fold benchmark speedup and relative error 0.00061.

Other slower experiments: sixteen query rows per SIMD group with 64 by 32 tiles, 8.55 milliseconds; vectorized four-element staging, 7.43 milliseconds; preconverting keys and values with PyTorch, 8.80 milliseconds; reusing query shared storage for keys, 7.77 milliseconds.

Half precision output accumulation is unsafe: repeated fresh-input checks produced errors up to 0.00364, exceeding the 0.002 tolerance. Retain float32 accumulation.
