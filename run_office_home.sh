#!/bin/bash
# sbatch run_a_c.sh
# sbatch run_a_p.sh
# sbatch run_a_r.sh

sbatch run_c_a.sh
sbatch run_c_p.sh
sbatch run_c_r.sh

sbatch run_p_r.sh
sbatch run_p_c.sh
sbatch run_p_a.sh

sbatch run_r_a.sh
sbatch run_r_c.sh
sbatch run_r_p.sh
