import numpy as np 
import pandas as pd
import glob
from scipy.ndimage import convolve
from scipy.spatial.distance import directed_hausdorff
from osgeo import gdal
from PIL import Image

edge_detection_kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]
sigma = 200



def calc_fit(realflow_path: str, simulflow_path: str) -> float:
    # jaccard_index(A, B) * exp(hausdorff_distance(A, B) / sigma^2)
  
    ji = jaccard_index(realflow_path, simulflow_path)    

    real_frontier = get_frontier(realflow_path, "data/real_frontier.jpg")
    simul_frontier = get_frontier(simulflow_path, "data/simul_frontier.jpg")
    h_distance = max(directed_hausdorff(real_frontier, simul_frontier)[0], directed_hausdorff(simul_frontier, real_frontier)[0])
    
    fit = ji * np.exp(-h_distance / (sigma**2))

    return fit



def intersection_cardinality(realflow_path: str, simulflow_path) -> int:
    img_real = gdal.Open(realflow_path)
    band_real = np.array(img_real.GetRasterBand(1).ReadAsArray())
    band_real = (band_real > 1e-2).astype(int)
    
    img_simul = gdal.Open(simulflow_path)
    band_simul = np.array(img_simul.GetRasterBand(1).ReadAsArray())
    band_simul = (band_simul > 0.0).astype(int)

    intersection = np.logical_and(band_real, band_simul)
    intersection_cardinality = np.count_nonzero(intersection)
   
    return intersection_cardinality



def union_cardinality(realflow_path: str, simulflow_path) -> int:
    img_real = gdal.Open(realflow_path)
    band_real = np.array(img_real.GetRasterBand(1).ReadAsArray())
    band_real = (band_real > 1e-2).astype(int)

    img_simul = gdal.Open(simulflow_path)
    band_simul = np.array(img_simul.GetRasterBand(1).ReadAsArray())
    band_simul = (band_simul > 1e-2).astype(int)

    union = np.logical_or(band_real, band_simul)
    union_cardinality = np.count_nonzero(union)

    return union_cardinality



def jaccard_index(realflow_path: str, simulflow_path: str) -> float:
    intersection_card = intersection_cardinality(realflow_path, simulflow_path)
    union_card = union_cardinality(realflow_path, simulflow_path)

    return intersection_card / union_card



def get_frontier(img_path: str, output_path: str) -> list:
    img = gdal.Open(img_path)
    band1 = np.array(img.GetRasterBand(1).ReadAsArray())
    band1 = (band1 > 1e-2).astype(int)

    #print(img_path, np.unique(band1), band1.shape)
    edges = convolve(band1, edge_detection_kernel)
    edges = (edges > 0.0).astype(np.uint8)

    img_array = 255 * edges 
    img_edges = Image.fromarray(img_array)
    img_edges = img_edges.convert('L')
    img_edges.save(output_path)
    
    edges_coord = np.argwhere(edges == 1)
    #print(edges_coord)

    return np.array(edges_coord)


print(calc_fit("data/lava-2017.bsq", "data/simulated.bsq"))
