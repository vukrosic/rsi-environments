General advice on making GPU kernels fast on an Apple M-series Mac with PyTorch on MPS.

## Measuring

Time only with the provided benchmark. Other jobs may share the GPU, so a single timing can be off by ten percent or more; treat small differences as noise and repeat a promising change before building on it. Keep a short log of every version you time, what changed and the result, so you can go back to the fastest one. Before you finish, make sure the file you leave behind is the fastest correct version, not the last experiment.

Check correctness after every change, on fresh inputs, before spending a timed run. A wrong answer scores nothing, and errors that sit just under the tolerance on one input can exceed it on another. Leave a safety margin on the tolerance.

## Where the time goes

First work out whether the operation is limited by memory traffic or by arithmetic. Count the bytes that must be read and written and the number of floating-point operations, and compare with the machine: an M4 GPU reads memory at roughly one hundred gigabytes per second and does a few trillion float32 operations per second, more in half precision. If the time is far above both limits, the cost is overhead: kernel launches, synchronization, temporary tensors, or a poor memory access pattern.

Each separate PyTorch operation launches at least one GPU kernel and usually writes a full temporary tensor to memory. A chain of small elementwise operations therefore reads and writes the same data many times. Removing temporaries and launches is often the first large win.

## Cheap changes first

Try the library's fused functions before writing your own code. Combined multiply-add matrix functions, in-place operations and preallocated outputs avoid temporaries. Reorder arithmetic so that scalar factors are applied to the smallest tensor. Avoid Python loops over elements or rows, and avoid moving data to the CPU or synchronizing in the middle of the computation.

Lower precision is often the next large win. Half precision halves memory traffic and can double matrix throughput. Keep accumulations, reductions and anything sensitive to rounding in float32, and check the error carefully; convert once at the start and once at the end rather than repeatedly.

Batched matrix functions are highly optimized on this hardware. Before replacing them with custom code, check how close they already are to the arithmetic limit; a custom kernel only pays off if it removes memory traffic or launches that the library cannot.

## Custom Metal shaders

PyTorch can compile a Metal shader at import time and run it on MPS tensors. Compile once at module level, never inside the timed function. A custom shader lets you fuse several steps into one pass over the data, keep intermediate values in registers or threadgroup memory instead of writing them to global memory, and choose your own work layout.

Design the work split before writing code. Decide what one thread, one SIMD group of thirty-two threads, and one threadgroup each compute. Give every thread enough independent work to hide memory latency, but not so much that registers spill. Common threadgroup sizes are 128 to 256 threads; try a few.

Memory access matters as much as arithmetic. Neighbouring threads should read neighbouring addresses so loads combine. Vector loads of four elements at a time reduce the number of memory instructions. Stage data that several threads reuse in threadgroup memory, load it cooperatively, and synchronize only where needed. Threadgroup memory is small, about thirty-two kilobytes, so tile large matrices.

For matrix products, tile the output: each threadgroup computes one block of the result, loops over the shared dimension in slices, loads a slice of each input into threadgroup memory, and accumulates in registers. Metal's SIMD-group matrix types multiply small eight-by-eight blocks in hardware and are much faster than scalar loops; use them for the inner product when the shapes allow. Unroll inner loops with fixed sizes so the compiler can keep values in registers.

For reductions such as sums and maxima, reduce first within a thread, then within the SIMD group using shuffle operations, and only then across SIMD groups through threadgroup memory. Avoid atomic operations on global memory in hot loops.

Handle edges explicitly when sizes are not multiples of the tile size, and keep branches that differ between neighbouring threads out of inner loops.

## Working order

Start from the simplest correct fast version and change one thing per timed run, so each result tells you something. When a change is slower, undo it rather than stacking more changes on top. Larger tiles, more threads and more unrolling are not always faster; register pressure and occupancy trade off, so measure. Keep the numerically safe version as a fallback while you experiment with riskier precision choices.

If a custom kernel is slower than the library, look for the reason before abandoning the idea: uncoalesced loads, too little work per thread, excessive synchronization, or scalar arithmetic where SIMD-group matrix operations would apply. Often one of these explains most of the gap.

## Before finishing

Run the correctness check one more time on the final file, make sure it does not read any file or depend on state from earlier calls, and confirm that the function's name, arguments and output shape match the reference. Remove debugging code and any synchronization that is not needed for correctness.

## Common mistakes

Compiling a shader or allocating large buffers inside the timed function. Caching results between calls; inputs change every call. Comparing timings taken under different load. Keeping an experiment that was faster once by chance. Reducing precision in an accumulation that then drifts past the tolerance on some inputs. Writing a custom kernel for a step that takes a small share of the total time while the large step is left untouched. Spending all timed runs on small parameter changes before trying a structurally different approach.

Profile where the time goes before deciding what to rewrite: time each stage separately by removing the others in a scratch copy, using the correctness check rather than timed runs. A stage that takes most of the time is where structural changes pay off; tuning the rest rarely moves the total by much.

Read the reference carefully. Some steps can be reordered, merged or computed on a smaller tensor without changing the result, for example applying a scale before or after a product, or computing a symmetric product once. Check any such rewrite against the reference before timing it.

Think about how many times each tensor is read from and written to global memory in your current version, and how few times it would need to be in the ideal version. The gap between those two numbers is a good guide to how much time can still be saved and where.

When several matrices are processed independently, process them in one launch rather than one by one, and give each threadgroup a fixed piece of one matrix so all of them run at once. Keep the intermediate results of one iteration in the layout the next iteration needs, so no transposes or copies are needed between steps.
