#!/usr/bin/zsh -l

if [[ "$1" == "slurm" ]]; then
    home="/data/home/jkataok1"
    # # VITS16 on Office31
    # sbatch train_avatar.sh vits16 amazon webcam 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vits16 amazon dslr 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vits16 dslr webcam 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vits16 webcam amazon 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vits16 dslr amazon 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vits16 webcam dslr 31 $home/AVATAR/data/datasets/office31 200 32 slurm

    # # VITS8 on Office31
    # sbatch train_avatar.sh vits8 amazon webcam 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vits8 amazon dslr 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vits8 dslr webcam 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vits8 webcam amazon 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vits8 dslr amazon 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vits8 webcam dslr 31 $home/AVATAR/data/datasets/office31 200 32 slurm

    # VITB16 on Office31
    # sbatch train_avatar.sh vitb16 amazon webcam 31 $home/AVATAR/data/datasets/office31 200 8 slurm
    sbatch train_avatar.sh vitb16 amazon dslr 31 $home/AVATAR/data/datasets/office31 200 8 slurm
    sbatch train_avatar.sh vitb16 dslr webcam 31 $home/AVATAR/data/datasets/office31 200 8 slurm
    sbatch train_avatar.sh vitb16 webcam amazon 31 $home/AVATAR/data/datasets/office31 200 8 slurm
    sbatch train_avatar.sh vitb16 dslr amazon 31 $home/AVATAR/data/datasets/office31 200 8 slurm 
    sbatch train_avatar.sh vitb16 webcam dslr 31 $home/AVATAR/data/datasets/office31 200 8 slurm

    # VITB8 on Office31
    # sbatch train_avatar.sh vitb8 amazon webcam 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vitb8 amazon dslr 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vitb8 dslr webcam 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vitb8 webcam amazon 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vitb8 dslr amazon 31 $home/AVATAR/data/datasets/office31 200 32 slurm
    # sbatch train_avatar.sh vitb8 webcam dslr 31 $home/AVATAR/data/datasets/office31 200 32 slurm

    # # VITS16 on office_home
    # sbatch train_avatar.sh vits16 art clipart 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 art product 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 art real_world 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 clipart product 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 clipart real_world 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 clipart art 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 product real_world 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 product art 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 product clipart 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 real_world art 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 real_world clipart 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vits16 real_world product 65 $home/AVATAR/data/datasets/office_home 200 16 slurm

    # # VITS8 on office_home
    # sbatch train_avatar.sh vits8 art clipart 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 art product 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 art real_world 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 clipart product 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 clipart real_world 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 clipart art 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 product real_world 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 product art 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 product clipart 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 real_world art 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 real_world clipart 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vits8 real_world product 65 $home/AVATAR/data/datasets/office_home 200 32 slurm

    # VITB16 on office_home
    # sbatch train_avatar.sh vitb16 art clipart 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 art product 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 art real_world 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 clipart product 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 clipart real_world 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 clipart art 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 product real_world 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 product art 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 product clipart 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 real_world art 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 real_world clipart 65 $home/AVATAR/data/datasets/office_home 200 16 slurm
    # sbatch train_avatar.sh vitb16 real_world product 65 $home/AVATAR/data/datasets/office_home 200 16 slurm

    # VITB8 on office_home
    # sbatch train_avatar.sh vitb8 art clipart 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 art product 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 art real_world 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 clipart product 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 clipart real_world 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 clipart art 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 product real_world 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 product art 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 product clipart 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 real_world art 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 real_world clipart 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
    # sbatch train_avatar.sh vitb8 real_world product 65 $home/AVATAR/data/datasets/office_home 200 32 slurm
else
    home="$home"
    # # VITS16 on Office31
    # source train_avatar.sh vits16 amazon webcam 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vits16 amazon dslr 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vits16 dslr webcam 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vits16 webcam amazon 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vits16 dslr amazon 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vits16 webcam dslr 31 $home/AVATAR/data/datasets/office31 200 32

    # # VITS8 on Office31
    # source train_avatar.sh vits8 amazon webcam 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vits8 amazon dslr 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vits8 dslr webcam 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vits8 webcam amazon 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vits8 dslr amazon 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vits8 webcam dslr 31 $home/AVATAR/data/datasets/office31 200 32

    # VITB16 on Office31
    # source train_avatar.sh vitb16 amazon webcam 31 $home/AVATAR/data/datasets/office31 200 18
    # source train_avatar.sh vitb16 amazon dslr 31 $home/AVATAR/data/datasets/office31 200 18
    # source train_avatar.sh vitb16 dslr webcam 31 $home/AVATAR/data/datasets/office31 200 18
    # source train_avatar.sh vitb16 webcam amazon 31 $home/AVATAR/data/datasets/office31 200 18
    # source train_avatar.sh vitb16 dslr amazon 31 $home/AVATAR/data/datasets/office31 200 18 
    # source train_avatar.sh vitb16 webcam dslr 31 $home/AVATAR/data/datasets/office31 200 18 

    # VITB8 on Office31
    # source train_avatar.sh vitb8 amazon webcam 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vitb8 amazon dslr 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vitb8 dslr webcam 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vitb8 webcam amazon 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vitb8 dslr amazon 31 $home/AVATAR/data/datasets/office31 200 32
    # source train_avatar.sh vitb8 webcam dslr 31 $home/AVATAR/data/datasets/office31 200 32

    # # VITS16 on office_home
    # source train_avatar.sh vits16 art clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 art product 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 art real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 clipart product 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 clipart real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 clipart art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 product real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 product art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 product clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 real_world art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 real_world clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits16 real_world product 65 $home/AVATAR/data/datasets/office_home 200 32

    # # VITS8 on office_home
    # source train_avatar.sh vits8 art clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 art product 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 art real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 clipart product 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 clipart real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 clipart art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 product real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 product art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 product clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 real_world art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 real_world clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vits8 real_world product 65 $home/AVATAR/data/datasets/office_home 200 32

    # VITB16 on office_home
    # source train_avatar.sh vitb16 art clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 art product 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 art real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 clipart product 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 clipart real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 clipart art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 product real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 product art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 product clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 real_world art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 real_world clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb16 real_world product 65 $home/AVATAR/data/datasets/office_home 200 32

    # VITB8 on office_home
    # source train_avatar.sh vitb8 art clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 art product 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 art real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 clipart product 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 clipart real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 clipart art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 product real_world 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 product art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 product clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 real_world art 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 real_world clipart 65 $home/AVATAR/data/datasets/office_home 200 32
    # source train_avatar.sh vitb8 real_world product 65 $home/AVATAR/data/datasets/office_home 200 32
fi