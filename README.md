# AutoBZOC.jl

This Julia project provides the code used in our preprint on optical
conductivity integration: [High-order and adaptive optical conductivity
calculations using Wannier interpolation. Lorenzo Van Muñoz, Jason Kaye, Alex
Barnett and Sophie Beck](https://arxiv.org/abs/2406.15466).

## Running the code

Instantiate the Julia project in this repository and then run each of the
scripts separately.
```bash
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate()'
julia --project fig1.jl > fig1.log
julia --project fig2.jl > fig2.log
julia --project fig3.jl > fig3.log
julia --project fig4.jl > fig4.log # these logs contain the data for Table 1
julia --project --threads `nproc` fig5.jl > fig5.log # Recommended to run this in parallel
```
Each of the scripts has been set to an easier set of parameters so that
validating that the script runs can be done before performing the calculations
in the paper. The parameters used in the paper are commented and can be
uncommented to use them. The results from the 'warmup' will be saved and reused.
Figures will be saved to files in a `figs_t2g` directory.

The final figures 1-4 could each take multiple hours to run.
Figure 5 may take a week on a node of a cluster.