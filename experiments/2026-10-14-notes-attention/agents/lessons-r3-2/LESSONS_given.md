Notes from the previous round of agents, one section per agent.

## Agent 1

The workload is contiguous float32 causal attention: batch eight, eight heads, sequence length 1024, head size 64. Only benchmark through bench.py; shared GPU load varies. This round used all ten runs; reference times ranged from 66.89 to 73.63 milliseconds.

The final kernel measured 7.08 milliseconds against 72.30 milliseconds, a 10.21 times speedup. Benchmark relative error was 0.00068 against the 0.002 tolerance.

Use a fused Metal shader compiled once at import. The final configuration processes 128 queries and 64 keys per tile with 512 threads. Each SIMD group handles eight query rows. Load queries directly from float inputs into half matrix registers and retain them across key tiles. Cooperatively stage keys and values as half in shared memory. Accumulate matrix products and outputs in float32. Store scaled scores as half, overwrite them with half probabilities, and maintain stable online softmax maxima and denominators in float32. Skip wholly future tiles.

Measured alternatives, in milliseconds: direct half buffers with shared float scores, 7.64; register scores and probabilities, 12.42; staging inputs with register scores, 12.13; shared half scores with 64 by 64 tiles, 7.29; 32 by 32 tiles, 7.70; unrolled output matrix loops, 7.16; float scores with query storage reused for values, 10.20; additionally unrolled reduction loops, 7.24; 128 by 32 tiles, 7.13.

Earlier rounds reported built-in causal attention around 17 milliseconds in float32 and 10.5 milliseconds with half inputs. A different fused float-score implementation reached 4.99 milliseconds. Preserve float32 matrix accumulation: half dot-product accumulation and repeated half output accumulation previously exceeded tolerance. Half score storage after float accumulation passed comfortably here. Avoid judging tiny timing differences confidently.

## Agent 2

The workload is float32 causal attention on MPS: eight batches, eight heads, 1024 positions, and 64 channels. The reported unchanged baseline is 67.1 milliseconds. GPU contention varies reference times; compare absolute kernel times too. Earlier rounds measured built-in causal attention around 17 milliseconds, or 10.5 to 11.7 milliseconds after converting inputs to half. Earlier custom fused implementations reached 4.99 milliseconds under different conditions.

This round used ten benchmarks. The final kernel measured 6.09 milliseconds against a 66.92 millisecond reference, giving 10.99 times speedup. Benchmark relative error was 0.0013. Twenty fresh-input checks passed, with maximum error 0.00121 versus the 0.002 limit.

The successful fused Metal shader uses 64 queries and 64 keys per tile, eight SIMD groups, and 256 threads. Compile once at import. Stage queries in shared half storage, load query matrices into registers, then reuse that storage for keys and values. Accumulate dot products in float32 but store scores in half. Scores and probabilities share one half buffer. Read each SIMD group’s scores completely before overwriting them. This reduces shared storage to sixteen kilobytes. Keep stable online softmax maxima, denominators, and cumulative output in float32. Accumulate value products in half over sixteen keys, then add to float32 output. Skip future key tiles.

Measured alternatives, in milliseconds: separate float scores and half probabilities, 9.15; all scores and probabilities in registers, 11.84; explicit unrolling, 11.64; register-only 32 by 32 tiles, 13.35; shared queries, 9.93; 32-key tiles with float value products, 9.68; sixteen-key half value products, 8.78; smaller key tiles, 8.49; half shared scores plus direct query register initialization and 32-key tiles, 9.55.

Earlier lessons remain: half dot-product accumulation can fail tolerance; long half output accumulation is unsafe. Row reductions require shuffle distances one and eight.

## Agent 3

The target is contiguous float32 causal attention: eight batches, eight heads, 1024 positions, and 64 features. The supplied baseline was 67.1 milliseconds. GPU sharing changes timings; compare absolute kernel times alongside speedups.

The final kernel took 4.10 milliseconds against a 71.10 millisecond reference, giving 17.36 times speedup. Benchmark relative error was 0.00061, below 0.002. All ten submissions passed.

Compile a fused Metal shader once. Use 64 queries and 64 keys per tile, with eight SIMD groups and 256 threads. Each SIMD group handles eight query rows. Keep queries, scores, probabilities, and partial outputs in registers. Use half matrix operands and probabilities, but float32 scores, output accumulation, running maxima, and denominators. Apply stable online softmax and skip future key tiles. Metal matrix rows span four lanes; exclusive-or shuffles of one and eight reduce each row. Explicitly unroll constant matrix loops. Stage inputs through four-element vector loads. Separate key and value buffers remove barriers, at a small measured benefit.

This round measured, in milliseconds: shared scores and probabilities, 7.59; register matrices without explicit unrolling, 9.15; that version with 32 queries, 11.47; fully unrolled 64-by-64 tiles, 5.20; shared matrices with four-lane softmax, 7.87; unrolled 32-key tiles, 5.57; 128-key tiles, 5.54; vector staging, 4.13; reordered matrix operations, 5.41; separate value staging, 4.10.

Earlier rounds found built-in attention around 17 milliseconds in float32 and 10.5–11.7 with half inputs. Their fastest fused variant reached 4.99 milliseconds. Preconverting inputs and larger query tiles generally regressed. Full half score or output accumulation can exceed tolerance; retain float32 accumulation. Vector staging improved this unrolled register implementation despite being slower in an earlier shared implementation.

## Agent 4

The workload is contiguous float32 causal attention: batch eight, eight heads, 1024 positions, head size 64. The supplied baseline was 67.1 milliseconds. GPU contention varies reference times; compare absolute kernel times and treat small differences cautiously.

Earlier rounds found built-in causal attention around 17 milliseconds in float32 and 10.5 to 11.7 milliseconds with half inputs. Fused Metal attention avoids materializing scores. Earlier best float32 accumulation measured 4.99 milliseconds. Compile shaders once at import.

This round used all ten benchmarks. A 64 query by 64 key shader with shared float32 scores took 8.29 milliseconds. Keeping scores and probabilities in registers without explicit unrolling worsened this to 11.86. Fully unrolling constant matrix loops reduced it to 4.38. With unrolling, 64 queries and 32 keys took 4.40; 32 queries and 64 keys took 5.86; 128 queries and 64 keys took 4.95 during a busy GPU run. A separate vectorized input conversion pass took 4.81.

Sixteen-key half precision value-product partial sums, added into float32 running accumulators, took 4.24 milliseconds. Vectorized four-element staging took 4.25. Applying causal element masks only on the diagonal tile improved this to 3.88 milliseconds versus a 68.38 millisecond reference, a 17.60 times speedup. This is the final implementation.

Use eight SIMD groups, 256 threads, shared half key/value tiles, register queries and probabilities, float32 scores, stable float32 online softmax, and float32 running output. Skip future key tiles. On M4, matrix lanes own two adjacent columns; row reductions use lane shuffle offsets one and eight.

Long half precision accumulation is unsafe. Restrict it to sixteen-key partial sums. Thirty fresh-input checks passed; the final twenty had maximum relative error 0.00114, below 0.002. Benchmark error was 0.0012.
