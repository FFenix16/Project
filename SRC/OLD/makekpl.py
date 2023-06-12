from PIL import Image

# Open the image file
image = Image.open('img/ChampImg/GarenSquare.png')

# Create a loop to save the image 50 times
for i in range(50):
    # Copy the image
    new_image = image.copy()

    # Save the new image with a unique name in the specified directory
    new_image.save(f'img/Part2/img/image_G{i}.png')
