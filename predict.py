from PIL import Image
from multi_level_classifier_net import ClassifierNet

if __name__ == "__main__":
    model = ClassifierNet()

    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue

        model.detect_image(image_1)
