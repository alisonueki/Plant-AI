import cv2

def getRGB(image_path):
    image = cv2.imread(image_path)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    mean_rgb = image_rgb.mean(axis=(0,1))
    return "Mean RGB values: " + str(mean_rgb)
