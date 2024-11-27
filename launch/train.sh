python train.py \
--images_folder '../../Dataset/Topographies/raw/FiguresStacked 8X8_4X4_2X2 Embossed' \
--label_path '../../Dataset/biology_data/TopoChip/MacrophageWithClass.csv' \
--dataset_name 'biological' \
--n_epochs 40000 \
--img_size 64 \
--batch_size 32 \
--num_workers 4 \
--train_dis_freq 1 \
--model_name 'ddpm_35m' \
# --n_classes 5 \