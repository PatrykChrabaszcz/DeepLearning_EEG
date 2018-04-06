#!/usr/bin/env bash

for ini_file in \
/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__17_38_07_833464/config.ini \
/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__13_31_06_483134/config.ini \
/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__16_36_23_030252/config.ini \
/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__11_27_32_675878/config.ini \
/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__12_29_17_057007/config.ini \
/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__18_39_50_900265/config.ini \
/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__15_34_37_260002/config.ini \
/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__10_25_48_955730/config.ini \
/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__14_32_51_759155/config.ini
do
echo 'Starting first'
python single_evaluation.py --ini_file $ini_file --validation_data_type test --forget_state 0 \
--balanced 0 --use_augmentation 0 \
--continuous 0 --random_mode 0
done

#1:


##2:
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__22_25_17_183230/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__03_47_11_406889/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__23_29_39_178102/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__21_20_55_663357/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__01_38_26_810918/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__19_12_13_458058/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__00_34_05_013458/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_26__20_16_32_785467/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__02_42_47_209556/config.ini

#3:
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__14_15_29_908819/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__10_06_09_680753/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__11_08_22_150314/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__09_01_16_533415/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__13_13_15_803305/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__16_20_20_016863/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__07_59_00_671589/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__15_17_54_328251/config.ini \
#/home/chrabasp/EEG_Results/BO_Anomaly_6/train_manager/2018_03_27__12_10_58_904353/config.ini