import os
import cv2
import numpy as np
from tqdm import tqdm
import concurrent.futures
from multiprocessing.pool import ThreadPool
import time
from PIL import Image


data_folder = os.path.join("..", "..",  "data", "yolo_police10k")
a = os.listdir(data_folder)
edges_foldername = "edges"
images_foldername = "images_stari"
depths_foldername = "depth_masks"
numpy_foldername = "numpy_matrices5dim"
depth_maks_extension = "-dpt_beit_large_512.png"
depth_to_grayscale = False

edges_path = os.path.join(data_folder, edges_foldername)
images_path = os.path.join(data_folder, images_foldername)
depths_path = os.path.join(data_folder, depths_foldername)
edges = os.listdir(edges_path)
images = os.listdir(images_path)
depths = os.listdir(depths_path)
edges.sort()
depths.sort()
images.sort()


def write_img_to_disk(imgID):
    edge_image = cv2.imread(os.path.join(edges_path, imgID))
    image = cv2.imread(os.path.join(images_path, imgID))
    depth_path = os.path.join(depths_path, imgID[:-4] + depth_maks_extension)
    depth_image = cv2.imread(depth_path)
    #convert depth and edges to grayscale
    if depth_to_grayscale:
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        depth_image = np.expand_dims(depth_image_gray, axis=2)
    edge_image_gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    edge_image_gray = np.expand_dims(edge_image_gray, axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert to RGB
    stacked = np.concatenate((image, edge_image_gray, depth_image), axis=2)
    np.save(os.path.join(data_folder, numpy_foldername, f"{imgID[:-4]}_7dimRGBCD.npy"), stacked)



#def jere_pad(img )


a = np.load(os.path.join(data_folder, numpy_foldername, "002ccf65cf100cc58438664b005482e5_5dimRGBCD.npy"))
a1 = np.copy(a)
a2 = np.copy(a)
#a_cv = cv2.imread(os.path.join(data_folder, numpy_foldername, "002ccf65cf100cc58438664b005482e5.jpg"))
b = a[:, :, 0:3]
b = cv2.cvtColor(b, cv2.COLOR_RGB2BGR)
color = (114, 114, 114)
color2= (114, 114, 114, 114, 114, 114, 114)
top, bottom, left, right = 0, 0, 100, 100
im = cv2.copyMakeBorder(b, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
a_2 = np.zeros((a.shape[0]+top+bottom, a.shape[1]+left+right, a.shape[2]), dtype=np.uint8)
n_2 = np.zeros((a.shape[0]+top+bottom, a.shape[1]+left+right, a.shape[2]), dtype=np.uint8)
for channel in range(a.shape[2]):
    a_2[:, :, channel] = cv2.copyMakeBorder(a1[:, :, channel], top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0, 0])
    n_2[:, :, channel] = np.pad(a2[:, :, channel], ((top, bottom), (left, right)), 'constant', constant_values=[[0, 0], [0, 0]])
#a_2[:,:,0:3] = cv2.cvtColor(a_2[:,:,0:3], cv2.COLOR_BGR2RGB)
n2 = n_2[:,:,0:3]
n2_canny = n_2[:,:,3:4]
n2_depth = n_2[:,:,4:5]
x2 = a_2[:,:,0:3]
x2_canny = a_2[:,:,3:4]
x2_depth = a_2[:,:,4:5]
x = a[:,:,0:3]
x_canny = a[:,:,3:4]
x_depth = a[:,:,4:5]
#im2 = cv2.copyMakeBorder(a, 0, 0, left, right, cv2.BORDER_CONSTANT, value=color2)
# add border
#b = cv2.cvtColor(b, cv2.COLOR_BGR2RGB)
c = Image.fromarray(b)
print(a.shape)

exit()
t1 = time.time()
with ThreadPool(20) as pool:
    pool.map(write_img_to_disk, images, chunksize=10)
t2=time.time()
print(f"gotov sa zapisom u {round(t2-t1,2)} sekundi")


"""
this loop stacks every image from every folder on top of itself
"""
for imageID in tqdm(images):
    #load all images independently
    edge_image = cv2.imread(os.path.join(edges_path, imageID))
    image = cv2.imread(os.path.join(images_path, imageID))
    depth_path = os.path.join(depths_path, imageID[:-4] + depth_maks_extension)
    depth_image = cv2.imread(depth_path)
    #convert depth and edges to grayscale
    if depth_to_grayscale:
        depth_image_gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        depth_image_gray = np.expand_dims(depth_image_gray, axis=2)
    edge_image_gray = cv2.cvtColor(edge_image, cv2.COLOR_BGR2GRAY)
    edge_image_gray = np.expand_dims(edge_image_gray, axis=2)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert to RGB
    #stack them all up in one big matrix
    stacked = np.concatenate((image, edge_image_gray, depth_image_gray), axis=2)
    #stacked = np.concatenate((image, edge_image_gray, depth_image), axis=2)
    map(write_img_to_disk, [stacked], [os.path.join(data_folder, numpy_foldername, f"{imageID[:-4]}_5dimensionalRGBCD.npy")])
    #np.save(os.path.join(data_folder, numpy_foldername, f"{imageID[:-4]}_7dimensionalRGBCD.npy"), stacked)
    pass
