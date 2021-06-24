import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb

from config import config
import numpy as np
import pandas as pd

import albumentations as A


class ImageRunner():
    
    def __init__(self, model, preprocess_input, img):

        self.image = img
        self.model = model
        self.preprocess_input = preprocess_input
    
    @staticmethod
    def denormalize(x):
        """Scale image to range 0..1 for correct plot"""
        x_max = np.percentile(x, 98)
        x_min = np.percentile(x, 2)    
        x = (x - x_min) / (x_max - x_min)
        x = x.clip(0, 1)
        return x
    
    @staticmethod
    def get_custom_test_augmentation():
        """Add paddings to make image shape divisible by 32"""
        test_transform = [
            A.Resize(384, 480),
            A.PadIfNeeded(384, 480)
        ]
        return A.Compose(test_transform)

    @staticmethod
    def get_preprocessing(preprocessing_fn):
        """Construct preprocessing transform

        Args:
            preprocessing_fn (callbale): data normalization function 
                (can be specific for each pretrained neural network)
        Return:
            transform: albumentations.Compose

        """

        _transform = [
            A.Lambda(image=preprocessing_fn),
        ]
        return A.Compose(_transform)

          
    def process_image(self):
        '''
        Classifies the shape of cloth
        '''
        agumentation = self.get_custom_test_augmentation()
        augmented_image = agumentation(image=self.image)['image']
        
        preprocessing = self.get_preprocessing(self.preprocess_input)
        processed_image = preprocessing(image=augmented_image)['image']
        
        processed_image = np.expand_dims(processed_image, axis=0)
        pr_mask = self.model.predict(processed_image)
        
        label_image = pr_mask.squeeze().argmax(axis=2)
        image_label_overlay = label2rgb(label_image, image=self.denormalize(processed_image.squeeze()), bg_label=0)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.imshow(image_label_overlay)
        text_colors = list(mcolors.CSS4_COLORS.keys())
        np.random.shuffle(text_colors)
        text_colors = iter(text_colors)

        for region in regionprops(label_image):
            # take regions with large enough areas
            if region.area >= 500:
                text_color = next(text_colors)
                # draw rectangle around segmented coins
                minr, minc, maxr, maxc = region.bbox
                rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                          fill=False, edgecolor=text_color, linewidth=2, label='Label', linestyle = 'dashed')
                ax.add_patch(rect)
                rx, ry = rect.get_xy()
                cx = rx + rect.get_width()/2.0
                cy = ry + rect.get_height()/2.0
                ax.annotate(config['all_classes'][region.label], (cx, cy), color=text_color, weight='bold', fontsize=14, ha='center', va='center')

        ax.set_axis_off()
        plt.tight_layout()
        plt.savefig("./static/result/result.png")
        return 'result.png'
        

            
            
    



