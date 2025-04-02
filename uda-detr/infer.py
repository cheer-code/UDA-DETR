import torch
import cv2 as cv

model = torch.hub.load('.' , 'custom'  , path = r"E:\AQT_weight\city_to_foggy\space_datr_map46.pth",
                       source="local")

img = cv.imread(r"E:\Coke\dataset\CityScapes_Foggy\leftImg8bit_foggy\test\berlin\berlin_000000_000019_leftImg8bit_foggy_beta_0.02.png")
result = model(img)
result.save()