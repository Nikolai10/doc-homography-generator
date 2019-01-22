class DataConfig:

    # dataset generation props
    target_dims = (384, 256)
    shift = 100
    mode_p = 1

    # augmentation props
    sigma=(0, 2.0)
    scale=0.05
    gamma=(0.5, 1.5)

    # producer
    mat_imgs = 'imgs_0.7.mat'
    mat_corners = 'corners_0.7.mat'
    key_imgs = 'all_imgs'
    key_corners = 'all_corners'