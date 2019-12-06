# Augmentation using albumentation
# https://albumentations.readthedocs.io/en/latest/examples.html
import albumentations as alb
import os, random, string
import cv2
from helpers.data_slicing import get_rand_name


def generate_aug_img(p=0.7, img_in='', mask_in='', img_out='', mask_out='', num_gen=0):

    pipeline = alb.Compose([
        alb.VerticalFlip(),
        alb.HorizontalFlip(),
        alb.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
        alb.ElasticTransform(border_mode=cv2.BORDER_REFLECT_101, alpha_affine=40, p=1),
        alb.OneOf([
            alb.GaussianBlur(p=0.7, blur_limit=3),
            alb.RandomRain(p=0.7, brightness_coefficient=0.6, drop_width=1, blur_value=5),
            alb.RandomSnow(p=0.7, brightness_coeff=1, snow_point_lower=0.3, snow_point_upper=0.5),
            alb.RandomShadow(p=0.6, num_shadows_lower=1, num_shadows_upper=1,
                             shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1)),
            alb.RandomFog(p=0.7, fog_coef_lower=0.7, fog_coef_upper=0.8, alpha_coef=0.1)
        ], p=0.8),
        alb.OneOf([
            alb.CLAHE(clip_limit=2),
            alb.IAASharpen(),
            alb.IAAEmboss(),
            alb.RandomBrightnessContrast(),
        ], p=0.6),
    ], p=p)

    # Apply pipeline for each image
    for _ in range(num_gen):
        # Shuffle out image list
        img_list = os.listdir(img_in)
        random.shuffle(img_list)
        index = random.randint(0, len(img_list) - 1)
        # Pick one image
        img_id = img_list[index]
        # Apply augmentation to the coosen image
        _img_in = cv2.imread(img_in + img_id)
        _mask_in = cv2.imread(mask_in + img_id)
        # Fit pipeline
        augmented = pipeline(image=_img_in, mask=_mask_in)
        # Get outcomes
        _img_out, _mask_out = augmented["image"], augmented["mask"]
        # Gen. out filename
        out_fname = get_rand_name()
        # Write file to out dir
        cv2.imwrite(img_out + out_fname + '.png', _img_out)
        cv2.imwrite(mask_out + out_fname + '.png', _mask_out)
