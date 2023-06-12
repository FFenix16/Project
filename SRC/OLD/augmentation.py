import albumentations as A
import cv2
import matplotlib.pyplot as plt
import os

# Load image
img = cv2.imread('img/sab/ZedSquare.png')

#Wherer to save

outPut_dir = 'img/final/'


# Define augmentation pipelines
pipeline1 = A.Compose([
    A.RandomCrop(width=120, height=120),
    A.HorizontalFlip(p=1.0),
    A.RandomBrightnessContrast(brightness_limit=(-0.5, 0.5), contrast_limit=(-0.5, 0.5), p=1.0),
])
pipeline2 = A.Compose([
    A.Resize(width=120, height=120),
    A.Blur(0.4),
    A.RandomGamma(gamma_limit=(0.8, 1.2), p=0.3),
])

pipeline3 = A.Compose([
    A.Rotate(limit=45, p=1.0),
    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, brightness_coeff=2, p=1.0),
    A.RandomFog(fog_coef_lower=0.3, fog_coef_upper=0.7, p=1.0),
])

pipeline4 = A.Compose([
    A.CLAHE(clip_limit=2.0, p=0.3),
    A.RandomSunFlare(flare_roi=(0, 0, 0.5, 0.5), num_flare_circles_lower=5, num_flare_circles_upper=10, p=0.3),
    A.RandomRain(slant_lower=-5, slant_upper=5, drop_length=10, drop_width=1, drop_color=(200, 200, 200), p=0.3),
])

pipeline5 = A.Compose([
    A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
    A.RandomShadow(shadow_roi=(0, 0, 1, 0.5), num_shadows_lower=1, num_shadows_upper=3, p=0.5),
    A.RandomBrightness(limit=0.2, p=0.5),
])

pipeline6 = A.Compose([
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
    A.RandomContrast(limit=0.2, p=0.5),
    A.RandomSizedCrop(min_max_height=(80, 120), height=120, width=120, p=0.5),
])

pipeline7 = A.Compose([
    A.Blur(blur_limit=(1, 3), p=0.5),
    A.ChannelShuffle(p=0.2),
    A.HueSaturationValue(hue_shift_limit=(-10, 10), sat_shift_limit=(-20, 20), val_shift_limit=(-10, 10), p=0.2),
])

pipeline8 = A.Compose([
    A.Blur(blur_limit=(1, 3), p=0.5),
    A.ChannelShuffle(p=0.2),
    A.HueSaturationValue(hue_shift_limit=(-10, 10), sat_shift_limit=(-20, 20), val_shift_limit=(-10, 10), p=0.2),
])
pipeline9 = A.Compose([
    A.Blur(blur_limit=(1, 3), p=0.5),
    A.ChannelShuffle(p=0.2),
    A.HueSaturationValue(hue_shift_limit=(-10, 10), sat_shift_limit=(-20, 20), val_shift_limit=(-10, 10), p=0.2),
])

# Apply augmentation pipelines to image
transformed1 = pipeline1(image=img)
transformed2 = pipeline2(image=img)
transformed3 = pipeline3(image=img)
transformed4 = pipeline4(image=img)
transformed5 = pipeline5(image=img)
transformed6 = pipeline6(image=img)
transformed7 = pipeline7(image=img)
transformed8 = pipeline8(image=img)
transformed9 = pipeline9(image=img)


# Display original and transformed images side by side
fig, ax = plt.subplots(2, 6)
ax[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
ax[0, 0].set_title('Original')
ax[0, 1].imshow(cv2.cvtColor(transformed1['image'], cv2.COLOR_BGR2RGB))
ax[0, 1].set_title('Transformed 1')
ax[0, 2].imshow(cv2.cvtColor(transformed2['image'], cv2.COLOR_BGR2RGB))
ax[0, 2].set_title('Transformed 2')
ax[0, 3].imshow(cv2.cvtColor(transformed3['image'], cv2.COLOR_BGR2RGB))
ax[0, 3].set_title('Transformed 3')
ax[0, 4].imshow(cv2.cvtColor(transformed4['image'], cv2.COLOR_BGR2RGB))
ax[0, 4].set_title('Transformed 4')
ax[0, 5].axis('off')  # Empty plot
ax[1, 0].imshow(cv2.cvtColor(transformed5['image'], cv2.COLOR_BGR2RGB))
ax[1, 0].set_title('Transformed 5')
ax[1, 1].imshow(cv2.cvtColor(transformed6['image'], cv2.COLOR_BGR2RGB))
ax[1, 1].set_title('Transformed 6')
ax[1, 2].imshow(cv2.cvtColor(transformed7['image'], cv2.COLOR_BGR2RGB))
ax[1, 2].set_title('Transformed 7')
ax[1, 3].imshow(cv2.cvtColor(transformed8['image'], cv2.COLOR_BGR2RGB))
ax[1, 3].set_title('Transformed 8')
ax[1, 4].imshow(cv2.cvtColor(transformed9['image'], cv2.COLOR_BGR2RGB))
ax[1, 4].set_title('Transformed 9')
ax[1, 5].axis('off')  # Empty plot

#Save images.
output_filename = 'ZedSquare.png'
cv2.imwrite(outPut_dir + output_filename, img)
output_filename1 = 'ZedSquare1.png'
cv2.imwrite(outPut_dir + output_filename1, transformed1['image'])
output_filename2 = 'ZedSquare2.png'
cv2.imwrite(outPut_dir + output_filename2, transformed2['image'])
output_filename3 = 'ZedSquare3.png'
cv2.imwrite(outPut_dir + output_filename3, transformed3['image'])
output_filename4 = 'ZedSquare4.png'
cv2.imwrite(outPut_dir + output_filename4, transformed4['image'])
output_filename5 = 'ZedSquare5.png'
cv2.imwrite(outPut_dir + output_filename5, transformed5['image'])
output_filename6 = 'ZedSquare6.png'
cv2.imwrite(outPut_dir + output_filename6, transformed6['image'])
output_filename7 = 'ZedSquare7.png'
cv2.imwrite(outPut_dir + output_filename7, transformed7['image'])
output_filename8 = 'ZedSquare8.png'
cv2.imwrite(outPut_dir + output_filename8, transformed8['image'])
output_filename9 = 'ZedSquare9.png'
cv2.imwrite(outPut_dir + output_filename9, transformed9['image'])
plt.tight_layout()
plt.show()
