class Config:

    # dataset generation props
    target_dims = (384, 256)
    shift = 100
    mat_imgs = 'imgs_25K.mat'
    mat_corners = 'corners_25K.mat'
    key_imgs = 'all_imgs'
    key_corners = 'all_corners'
    mode_p = 0.7

    # augmentation props
    sigma=(0, 2.0)
    scale=0.05
    gamma=(0.5, 1.5)