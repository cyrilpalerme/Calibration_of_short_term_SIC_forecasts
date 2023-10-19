#!/bin/bash -f
#$ -N Predictions_COSI
#$ -l h_rt=02:00:00
#$ -S /bin/bash
#$ -pe shmem-1 1
#$ -l h_rss=4G,mem_free=4G,h_data=12G
#$ -q gpu-r8.q
##$ -j y
#$ -m ba
#$ -o /home/cyrilp/Documents/OUT/OUT_$JOB_NAME.$JOB_ID_$TASK_ID
#$ -e /home/cyrilp/Documents/ERR/ERR_$JOB_NAME.$JOB_ID_$TASK_ID
##$ -R y
##$ -r y

module use /modules/MET/rhel8/user-modules/
module load cuda/11.6.0

source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate /lustre/storeB/users/cyrilp/mycondaTF

python3 "/lustre/storeB/users/cyrilp/COSI/Scripts/Models/SIC_Attention_Res_UNet_without_TOPAZ_SIC/Make_predictions_8.py"
rm /home/cyrilp/Documents/ERR/ERR_Predictions_COSI.* /home/cyrilp/Documents/OUT/OUT_Predictions_COSI.*
