#!/bin/bash
sbatch a_to_w_run.sh
sbatch a_to_d_run.sh
sbatch d_to_a_run.sh
sbatch w_to_a_run.sh
sbatch d_to_w_run.sh
sbatch w_to_d_run.sh