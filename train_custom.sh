python3 train.py \
--data_name custom --cnn_type resnext_wsl --wemb_type glove \
--margin 0.2 --max_violation --img_num_embeds 4 --txt_num_embeds 4 \
--img_attention --txt_attention --txt_finetune \
--batch_size 100 --warm_epoch 0 --num_epochs 30 \
--optimizer adamw --lr_scheduler cosine --lr_step_size 30 --lr_step_gamma 0.1 \
--warm_img --finetune_lr_lower 1 \
--lr 1e-4 --txt_lr_scale 1 --img_pie_lr_scale 0.1 --txt_pie_lr_scale 0.1 \
--eval_on_gpu --sync_bn --amp \
--temperature 16 \
--txt_pooling cls --arch slot --txt_attention_input wemb \
--spm_img_pos_enc_type none --spm_txt_pos_enc_type sine \
--spm_1x1 --spm_residual --spm_residual_norm --spm_residual_activation none \
--spm_activation gelu \
--spm_ff_mult 4 --spm_last_ln \
--img_res_pool max --img_res_first_fc \
--spm_input_dim 1024 --spm_query_dim 1024 \
--spm_depth 4 --spm_weight_sharing \
--remark custom_no_imagefineunte \
--res_only_norm --img_1x1_dropout 0.1 --spm_pre_norm \
--gpo_1x1 --gpo_rnn \
--weight_decay 1e-4 --grad_clip 1 --lr_warmup -1 --unif_residual \
--workers 8 --dropout 0.1 --caption_drop_prob 0.2 --butd_drop_prob 0.2 \
--image_root /content/drive/MyDrive/images --json_root /content/drive/MyDrive/data_temp/data_list.pickle \
--use_bert \
--mmd_weight 0. --unif_weight 0. \
--loss max --eval_similarity max
#--loss smooth_chamfer --eval_similarity smooth_chamfer
#--img_finetune 

#txt_pooling rnn -> cls