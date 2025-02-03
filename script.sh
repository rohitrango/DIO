conda activate deq-rieadam-reg
{ gpu 0 && python train_multi_level_2d.py --config-name oasis_ml_freeform_d4_2D exp_name=diceonly_oasis2d_freeform_sgdphantom diffopt.phantom_step=sgd loss.weight_ncc=0 } &
{ gpu 3 && python train_multi_level_2d.py --config-name oasis_ml_freeform_d4_2D exp_name=diceonly_oasis2d_freeform_adamphantom diffopt.phantom_step=adam loss.weight_ncc=0 } &
{ gpu 4 && python train_multi_level_2d.py --config-name oasis_ml_d4_2D exp_name=diceonly_oasis2d_diffeo_adamphantom num_workers=8 batch_size=16 diffopt.phantom_step=adam loss.weight_ncc=0 }& 
{ gpu 5 && python train_multi_level_2d.py --config-name oasis_ml_d4_2D exp_name=diceonly_oasis2d_diffeo_sgdphantom num_workers=8 batch_size=16 diffopt.phantom_step=sgd loss.weight_ncc=0 } &

python train_multi_level_3d.py --config-name oasis_ml_freeform_d4_3D exp_name=oasis3d_ff_shallow_sgd_p3 diffopt.phantom_step=sgd model.f_maps=[16,32,32,64,64]

python train_multi_level_3d.py --config-name oasis_ml_freeform_d4_3D exp_name=oasis3d_ff_sgd_p1_shallow_moreiters diffopt.iterations=[200,100,50] diffopt.phantom_step=sgd model.f_maps=[16,32,32,64,64] train.train_new_level=[5,30] loss.weight_ncc=0.5

python train_multi_level_3d.py --config-name oasis_ml_freeform_d4_3D exp_name=oasis3d_transmorph_ff_sgd_p1_default_moreiters diffopt.iterations=[200,100,50] diffopt.phantom_step=sgd train.train_new_level=[5,30] loss.weight_ncc=0.25 model.name=transmorph
python train_multi_level_3d.py --config-name oasis_ml_freeform_d4_3D exp_name=oasis_unsigned_ncc train.cc_unsigned=True

### model with unsigned ncc
python train_multi_level_3d.py --config-name oasis_ml_freeform_d4_3D exp_name=oasis_unsigned_ncc train.cc_unsigned=True
python train_multi_level_3d.py --config-name oasis_4x2x1x_unetencoder exp_name=oasis_enc_l2_contdiffeo loss.img_loss=mse resume=True train.epochs=500 model_path=saved_models/oasis_ml_4x2x1x/oasis_enc_l2/best_dice_loss.pth dataset.data_root=/mnt/rohit_data2/OASIS/ train.lr=1e-5 num_workers=2 diffopt.warp_type=diffeomorphic diffopt.phantom_step=sgd
python train_multi_level_3d.py --config-name oasis_4x2x1x_unetencoder exp_name=oasis_lku_l2 loss.img_loss=mse train.epochs=500 model.name=lku dataset.data_root=/mnt/rohit_data2/OASIS/
python train_multi_level_3d.py --config-name oasis_4x2x1x_unetencoder exp_name=oasis_lku_ncc loss.img_loss=ncc train.epochs=500 model.name=lku model.f_maps=[16]
python train_multi_level_3d.py --config-name oasis_4x2x1x_unetencoder exp_name=oasis_lku_l2_ddp loss.img_loss=mse train.epochs=500 model.name=lku model.f_maps=[16] ddp.enabled=True resume=True model_path=saved_models/oasis_ml_4x2x1x/oasis_lku_l2/best_dice_loss.pth train.lr=2e-4 train.lr_power_decay=0


### Test LKU on diffeomorphic transform (trained on SGD)
python test_multi_level_klein.py --config-path saved_models/oasis_ml_4x2x1x/oasis_lku_l2_ddp/ --config-name config.yaml +dry_run=True hydra.job.chdir=False diffopt.iterations=[400,200,100] diffopt.warp_type=diffeomorphic diffopt.phantom_step=adam +learn2reg_eval=False


### Test on OASIS (UNet)
python test_multi_level_3d.py --config-path saved_models/oasis_ml_4x2x1x/oasis_enc_l2_cont/ --config-name config.yaml +learn2reg_eval=False dataset.split=val hydra.job.chdir=False
python test_multi_level_3d.py --config-path saved_models/oasis_ml_ff_d4/oasis_unsigned_ncc/ --config-name config.yaml +learn2reg_eval=False dataset.split=val hydra.job.chdir=False
# LKU network
python test_multi_level_3d.py --config-path saved_models/oasis_ml_4x2x1x/oasis_lku_l2_ddp/
--config-name config.yaml +learn2reg_eval=False dataset.split=val hydra.job.chdir=False diffopt.warp_type=diffeomorphic diffopt.phantom_step=adam diffopt.iterations=[400,200,100]
# LKU encoder
python test_multi_level_3d.py --config-path saved_models/oasis_ml_4x2x1x/oasis_lkuencv2_mse/ --config-name config.yaml +learn2reg_eval=False dataset.split=val hydra.job.chdir=False diffopt.warp_type=diffeomorphic diffopt.phantom_step=adam diffopt.iterations=[400,200,100]

PYTHONPATH=./ python scripts/dio/train_multi_level_3d.py --config-name oasis_4x2x1x_unetencoder exp_name=oasis_lku_l2_per98_phantom5 loss.img_loss=mse train.epochs=100 model.name=lku dataset.data_root=/mnt/rohit_data2/OASIS_old/ +diffopt.learning_rate=3e-3 train.train_new_level=[5,20] save_every=50 loss.weight_dice=1.0 loss.dice_l2_mode=dice deploy=True loss.weight_ncc=1.0 diffopt.n_phantom_steps=5
PYTHONPATH=./ python scripts/dio/train_multi_level_3d.py --config-name oasis_4x2x1x_unetencoder exp_name=oasis_lku_l2_per98_phantom3 loss.img_loss=mse train.epochs=100 model.name=lku dataset.data_root=/mnt/rohit_data2/OASIS_old/ +diffopt.learning_rate=3e-3 train.train_new_level=[5,20] save_every=50 loss.weight_dice=1.0 loss.dice_l2_mode=dice deploy=True loss.weight_ncc=1.0 diffopt.n_phantom_steps=3
PYTHONPATH=./ python scripts/dio/train_multi_level_3d_kps.py exp_name=lungct_v1ncc_adam_unet diffopt.phantom_step=sgd/adam
