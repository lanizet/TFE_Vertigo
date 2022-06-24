#!/bin/bash
# Submission script for Lemaitre3
#SBATCH --time=00:10:00 # hh:mm:ss
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10000 # megabytes
# 
#SBATCH --mail-user=astrid.doyen@student.uclouvain.be
#SBATCH --mail-type=ALL
#
#SBATCH --output='/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_preprocessing/fmripreprocsmooth_out.txt'
#SBATCH --error='/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/functional_preprocessing//fmripreprocsmooth_err.txt'
#
#SBATCH --job-name=fMRI_preprocessing_smooth

line=$1
path='/CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/fMRI/preproc_smooth/' ;

#Load functional data and save time t=1
mkdir -p $path/mc_$line;
fslmaths /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset/PROJECT/Resting_state/$line.nii.gz $path/mc_$line/${line}_prefiltered_func_data -odt float
fslroi $path/mc_$line/${line}_prefiltered_func_data $path/mc_$line/${line}_example_func 1 1

#Motion correction
mkdir -p $path/mc_$line/motion;
mcflirt -in /CECI/proj/pilab/PermeableAccess/vertige_LEWuQhzYs9/ELIKOPY_subset_new/PROJECT/Resting_state/$line.nii.gz -out $path/motioncorrection_$line -mats -plots -rmsrel -rmsabs -spline_final;
mv -f $path/motioncorrection_$line.mat $path/motioncorrection_$line.par $path/motioncorrection_${line}_abs.rms $path/motioncorrection_${line}_abs_mean.rms $path/motioncorrection_${line}_rel.rms $path/motioncorrection_${line}_rel_mean.rms $path/motioncorrection_$line.nii.gz $path/mc_$line/motion ;
fsl_tsplot -i $path/mc_$line/motion/motioncorrection_$line.par -t 'MCFLIIRT estimated rotations (radians)' -u 1 --start=1 --finish=3 -a x,y,z -w 640 -h 144 -o $path/mc_$line/motion/${line}_rotation.png ;
fsl_tsplot -i $path/mc_$line/motion/motioncorrection_$line.par -t 'MCFLIIRT estimated translations (mm) (radians)' -u 1 --start=4 --finish=6 -a x,y,z -w 640 -h 144 -o $path/mc_$line/motion/${line}_translation.png ;
fsl_tsplot -i $path/mc_$line/motion/motioncorrection_${line}_abs.rms,$path/mc_$line/motion/motioncorrection_${line}_rel.rms -t 'MCFLIRT estimated mean displacement (mm)' -u 1 -w 640 -h 144 -a absolute,relative -o $path/mc_$line/motion/${line}_distance.png ;

#Slice timing correction
mkdir -p $path/mc_$line/time_correction;
slicetimer -i $path/mc_$line/motion/motioncorrection_$line.nii.gz -o $path/mc_$line/time_correction/slicetimer_$line.nii.gz -v -r 2 --odd ;

#Smoothing
fslmaths $path/mc_$line/time_correction/slicetimer_$line.nii.gz -Tmean $path/mc_$line/tmeansmooth_$line;
bet2 $path/mc_$line/tmeansmooth_$line mask -f 0.3 -n -m;
immv mask_mask mask;
fslmaths $path/mc_$line/time_correction/slicetimer_$line.nii.gz -mas mask $path/mc_$line/smoothbet_$line.nii.gz
fslstats $path/mc_$line/smoothbet_$line.nii.gz -p 2 -p 98 > $path/mc_$line/stats.txt;
var="$(cat $path/mc_$line/stats.txt)";
echo $var | cut -d' ' -f2 >> $path/mc_$line/stats2.txt;
scdvalue="$(cat $path/mc_$line/stats2.txt)";
thre=$(echo "scale=7 ; $scdvalue / 10" | bc) ;
fslmaths $path/mc_$line/smoothbet_$line.nii.gz -thr $thre -Tmin -bin mask -odt char; 
fslstats $path/mc_$line/time_correction/slicetimer_$line.nii.gz -k mask -p 50 > $path/mc_$line/stats_50.txt;
var2="$(cat $path/mc_$line/stats_50.txt)";
echo $var2 | cut -d' ' -f1 >> $path/mc_$line/stats2_50.txt;
scdvalue2="$(cat $path/mc_$line/stats2_50.txt)";
thre2=$(echo "scale=7 ; $scdvalue2 * 0.75" | bc) ;
fslmaths mask -dilF mask;
fslmaths $path/mc_$line/time_correction/slicetimer_$line.nii.gz -mas mask $path/mc_$line/presmooth;
fslmaths $path/mc_$line/presmooth -Tmean $path/mc_$line/tmeanmooth_$line;
susan $path/mc_$line/presmooth $thre2 1.698 3 1 1 $path/mc_$line/tmeansmooth_$line $thre2 $path/mc_$line/prefiltered_func_data_smooth_$line

#High pass filtering
mkdir -p $path/mc_$line/highpass;
fslmaths $path/mc_$line/prefiltered_func_data_smooth_$line.nii.gz -Tmean $path/mc_$line/highpass/tempMean_$line.nii.gz;
fslmaths $path/mc_$line/prefiltered_func_data_smooth_$line.nii.gz -bptf 50 -1 -add $path/mc_$line/highpass/tempMean_$line.nii.gz $path/mc_$line/highpass/highpass_$line.nii.gz;

printf "Successfully preprocessed patient %s \n" $line

