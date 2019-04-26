import albumentations as A


def augmentations(prob=0.95):
    
    transformer = A.Compose([

            A.OneOf([A.HorizontalFlip(p=prob),
                     A.VerticalFlip(p=prob)], p=prob),

            A.ShiftScaleRotate(p=prob, shift_limit=0.2, scale_limit=.2, rotate_limit=45),
            A.RandomRotate90(p=prob),
            A.Transpose(p=prob),
            A.OneOf([A.RandomContrast(limit=0.2, p=prob),
                     A.RandomGamma(gamma_limit=(70, 130), p=prob),
                     A.RandomBrightness(limit=0.2, p=prob)],p=prob),
            A.HueSaturationValue(p=prob),
            A.OneOf([
                    A.MotionBlur(p=prob),
                    A.MedianBlur(blur_limit=3, p=prob),
                    A.Blur(blur_limit=3, p=prob)
            ], p=prob),
            A.OpticalDistortion(p=prob),
            A.GridDistortion(p=prob),
            A.OneOf([
                    A.IAAAdditiveGaussianNoise(p=prob),
                    A.GaussNoise(p=prob),
                  ], p=prob),
    ], p=prob)
    return transformer


