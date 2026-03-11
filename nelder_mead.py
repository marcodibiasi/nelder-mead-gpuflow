import numpy as np 
import sys
import pandas as pd
import os
import glob
from scipy.ndimage import convolve
from scipy.spatial.distance import directed_hausdorff
from osgeo import gdal
from PIL import Image
import ast
import json

import setup_files as setup
import simulation_interface as simul


# utilities
init_path = "setup-data/init_curr.txt"
vent_path = "setup-data/vent_curr.txt"
real_flow_path = "data/lava-2017"
exclusion_path = "setup-data/exclude.txt"
real_frontier_path = "real_frontier.jpg"
simul_frontier_path = "simul_frontier.jpg"
edge_detection_kernel = [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]


# constants for the fit function (meters)
# 10% length of the lenght of the flow
sigma = 200


# error in measurement of easting and northing (meters)
location_error = 200


# fixed range for simulation variables
flux_pct_range = [0, 1.5]
h2o_range = [-2.0, -0.5]          #log10 of the real range


# the current version only works with this vent
vent_location = [500060, 4177705]


# vent.txt and init.txt constant data
t_start = 0
t_end = 249960
dt_save = 249960
emissivity = 0.9
fluxrate = "data/fluxrate-2017.txt"
dem_file = "data/dem_2016_corr_10m"

# space dimension
dim = 4


# n is how many latin hypercube should be generated, therefore the number of simulation is n*5
def simplex_scouting(n: int, ranges: list, masks: list, minimum_distance: int) -> list:
    samples = []
    scores = []

    for i in range(n): 
        latin_sample = latin_hypercube()
        for j in range(len(latin_sample)):
            denorm_vertex = denormalize_vertex(latin_sample[j], ranges)
            samples.append(latin_sample[j])

            simul_index = "s" + str(i*5 + j)
            scores.append(vertex_simulation(denorm_vertex, simul_index, masks))
    
    vertices_scores = zip(scores, samples, range(n*5))
    vertices_scores = sorted(vertices_scores, key = lambda x: x[0])
    sorted_scores, sorted_samples, sorted_indices = zip(*vertices_scores)
    
    final_simplex = []
    final_simplex.append(sorted_samples[0])
    final_scores.append(sorted_scores[0])
    
    curr_vertex = 1
    while curr_vertex < len(sorted_samples[1:]) and len(final_simlex) < 5: 
        if np.linalg.norm(sorted_samples[0] - sorted_samples[curr_vertex]) >= minimum_distance:
            final_simplex.append(sorted_samples[curr_vertex])
            final_scores.append(sorted_scores[curr_vertex])
        curr_vertex += 1

    for i in range(5): 
        original_path = "runs/run-s" + str(sorted_indices[i]) + "-dir/"
        rename_path = "runs/run-" + str(i) + "-dir/"
        os.rename(original_path, rename_path)

    return final_scores, final_simplex 



def generate_1d_grid(resolution: int) -> list:
    sample_size = 1 / resolution
    offset = 0.5 * sample_size

    return offset + np.random.permutation(np.linspace(0, 1, resolution, endpoint=False)) 



def generate_1d_grid_centered(resolution: int, grid_subdivision_padding: int) -> list:
    complete_resolution = resolution + grid_subdivision_padding

    # the inner offset is relative to the grid space and allows points to be centered 
    inner_offset = 0.5 / complete_resolution

    # the outer offset prevent having point too nearby the borders 
    outer_offset = (1 / complete_resolution) * grid_subdivision_padding

    return inner_offset + np.random.permutation(np.linspace(outer_offset, 1 - outer_offset, resolution, endpoint=False)) 



def latin_hypercube() -> list:
    # Generates a shuffled grid of dimension dim
    grid_subdivision_padding = 2
    #grid = [generate_1d_grid_centered(dim + 1, grid_subdivision_padding) for _ in range(dim)]
    grid = [generate_1d_grid(dim + 1) for _ in range(dim)]

    # Latin Hypercube by stacking the points 
    lhs_points = np.stack(grid, axis=0).T

    print(lhs_points)
    return lhs_points



def denormalize_vertex(point: list, ranges: list) -> list:
    denorm_vertex = []
    for i in range(dim):
        denorm_component = point[i] * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
        denorm_vertex.append(denorm_component) 

        if(denorm_component > ranges[i][1] or denorm_component < ranges[i][0]):
            print("Component out of range: " + str(i))

    return denorm_vertex
        


def denormalize_simplex(simplex: list, ranges: list) -> list:
    denorm_simplex = []
    for i in range(dim + 1):
        denorm_simplex.append(denormalize_vertex(simplex[i], ranges))

    return denorm_simplex



def vertex_simulation(vertex: list, simul_index: int, masks: tuple) -> float:
    init_file = setup.InitTemplate(emissivity, pow(10, vertex[2]), t_end, dt_save)
    vent_file = setup.VentTemplate(vertex[0], vertex[1], t_start, t_end, vertex[3], fluxrate)
    init_file.save(init_path)
    vent_file.save(vent_path)
    dir_path = "runs/run-" + str(simul_index) + "-dir/"
    
    print("Simulating vertex n.", simul_index)
    
    # Handling masks 
    if (in_mask(masks, vertex[:2])): 
        os.makedirs(dir_path, exist_ok= True)
        return 0

    simul.run_simulation(dem_file, init_path, vent_path, simul_index)
    
    simulflow_star = dir_path + "*final.bsq"    
    simulflow_path = glob.glob(simulflow_star)[0]
    
    resizedflow_path = dir_path + "/simulated.bsq"
    resize_raster(real_flow_path, simulflow_path, resizedflow_path)
    
    fit = -calc_fit(real_flow_path, resizedflow_path, simul_index) 

    s_vertex = f"{simul_index}: {vertex[0]}, {vertex[1]}, {vertex[2]}, {vertex[3]}"
    with open(dir_path + "vertex_info.txt", "w", encoding="utf-8") as f: 
        f.write(s_vertex)
        f.write(f"\n{fit}")

    export_geojson(dir_path + "coordinates.json", vertex[0], vertex[1])

    return fit


def export_geojson(filename: str, easting: int, northing: int) -> None:
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Point",
                    "coordinates": [easting, northing]
                }
            }
        ]
    }

    with open(filename, "w") as f:
        json.dump(geojson, f, indent=2)


def simplex_simulation(simplex: list, simul_index: int, masks: tuple) -> list:
    # the fuction takes an already denormalized simplex
    scores = []
    
    idx = simul_index
    for vertex in simplex:
        scores.append(vertex_simulation(vertex, idx, masks))
        idx += 1

    return scores



def calc_fit(realflow_path: str, simulflow_path: str, simul_index: int) -> float:
    # jaccard_index(A, B) * exp(hausdorff_distance(A, B) / sigma^2)
  
    ji = jaccard_index(realflow_path, simulflow_path)    

    simulfrontier_path = "runs/run-" + str(simul_index) + "-dir/" + simul_frontier_path
    real_frontier = get_frontier(realflow_path, real_frontier_path)
    simul_frontier = get_frontier(simulflow_path, simulfrontier_path)
    h_distance = max(directed_hausdorff(real_frontier, simul_frontier)[0], directed_hausdorff(simul_frontier, real_frontier)[0])
    #print(h_distance, np.exp(-h_distance / sigma**2), ji)
    
    fit = ji * np.exp(-h_distance / (sigma**2))
    #print(fit)

    return fit



def resize_raster(source_path: str, target_path: str, output_path: str) -> None:
    reference = gdal.Open(source_path)    
    gt = reference.GetGeoTransform()
    reference_projection = reference.GetProjection()
    reference_width = reference.RasterXSize
    reference_height = reference.RasterYSize

    gdal.Warp(
        output_path,
        target_path,
        format = 'ENVI',
        width = reference_width,
        height = reference_height,
        outputBounds = (gt[0],
                        gt[3] + reference_height * gt[5],
                        gt[0] + reference_width * gt[1],
                        gt[3]),
        dstSRS = reference_projection,
        resampleAlg = 'near',
        srcNodata = 0,
        dstNodata = -9999
    )



def jaccard_index(realflow_path: str, simulflow_path: str) -> float:
    img_real = gdal.Open(realflow_path)
    band_real = np.array(img_real.GetRasterBand(1).ReadAsArray())
    band_real = (band_real > 0.0).astype(int)
    
    img_simul = gdal.Open(simulflow_path)
    band_simul = np.array(img_simul.GetRasterBand(1).ReadAsArray())
    band_simul = (band_simul > 0.0).astype(int)

    intersection = np.logical_and(band_real, band_simul)
    intersection_card = np.count_nonzero(intersection)

    union = np.logical_or(band_real, band_simul)
    union_card = np.count_nonzero(union) 

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



def get_centroid(simplex: list) -> list:
    return sum(simplex) / (dim + 1)


def order(simplex: list, simplex_scores: list) -> list:

    # sort the vertices from best to worst fit
    vertices_scores = zip(simplex_scores, simplex)
    vertices_scores = sorted(vertices_scores, key = lambda x: x[0])
    return vertices_scores 
    


def shrink(best_point: list, points: list, sigma: int, simul_index: int, ranges: list, masks: list) -> list:
    simplex = []
    scores = []

    for i in range(len(points)): 
        vertex = best_point + sigma * (points[i] - best_point)
        denorm_vertex = denormalize_vertex(vertex, ranges)
        score = vertex_simulation(denorm_vertex, simul_index, masks)

        simplex.append(vertex)
        scores.append(score)
        simul_index+=1

    return simplex, scores, simul_index



def get_masks(path: str) -> list:
    masks = []

    with open(path, "r") as f: 
        for row in f: 
            row = row.strip()
            if row: 
                masks.append(ast.literal_eval(row))
    return masks


def in_mask(masks: tuple, vertex: list) -> bool:
    for mask in masks: 
        if((vertex[0] - mask[0][0])**2 + (vertex[1] - mask[0][1])**2 <= mask[1]**2):
            print("Vertex inside a mask")
            return True
    print("Vertex outside masks")
    return False


# !! EASTING NORTHING H20 FLUXPCT !!
def nelder_mead(ranges: list) -> None: 
    simul_index = dim + 1
    masks = get_masks(exclusion_path)

    # STARTING STEP: GENERATE AND EVALUATE THE STARTING SIMPLEX
    # simplex = latin_hypercube()
    # denorm_simplex = denormalize_simplex(simplex, ranges)
    # simplex_scores = simplex_simulation(denorm_simplex, 0, masks)
    simplex, simplex_scores = simplex_scouting(2, ranges, masks, 0.3) 
    print(simplex_scores)

    # operation coefficients
    alpha = 1.0     # reflection coefficient
    gamma = 2.0     # expansion coefficient
    rho = 0.5       # contraction coefficient
    sigma = 0.5     # shrink coefficient

    # stopping condition
    epsilon = 1e-3

    # MAIN LOOP
    while True:
        print("ORDER")

        # FIRST STEP: ORDER
        simplex_scores, sorted_vertices = zip(*order(simplex, simplex_scores))
        simplex_scores = list(simplex_scores)
        sorted_vertices = list(sorted_vertices)

        print(max(simplex_scores) - min(simplex_scores))
        if max(simplex_scores) - min(simplex_scores) < epsilon:
            break

        print(simplex_scores)

        # SECOND STEP: CENTROID
        print("CENTROID")
        centroid = get_centroid(sorted_vertices[: -1])
        

        # THIRD STEP: REFLECTION
        print("REFLECTION")

        # reflect the worst scoring vertex across the centroid
        reflected_vertex = centroid + alpha * (centroid - sorted_vertices[-1])
        
        # evaluate the reflected vertex
        denorm_reflected = denormalize_vertex(reflected_vertex, ranges)
        reflected_score = vertex_simulation(denorm_reflected, simul_index, masks)
        simul_index += 1

        best = simplex_scores[0]
        second_worst = simplex_scores[-2]
        worst = simplex_scores[-1]

        if best <= reflected_score < second_worst:  
            simplex_scores[-1] = reflected_score
            sorted_vertices[-1] = reflected_vertex
            print("score(x_1) <= score(x_r) < score(x_n))", best, "<=", reflected_score, " < ", second_worst)
            # back to first step: order

        elif reflected_score < best: 
            print("EXPANSION")

            expanded_vertex = centroid + gamma * (reflected_vertex - centroid)
            denorm_expanded = denormalize_vertex(expanded_vertex, ranges)
            expanded_score = vertex_simulation(denorm_expanded, simul_index, masks)
            simul_index += 1

            if expanded_score < reflected_score:
                simplex_scores[-1] = expanded_score
                sorted_vertices[-1] = expanded_vertex
                print("score(x_e) < score(x_r)", expanded_score, " < ", reflected_score)
                # back to first step: order
            
            else: 
                simplex_scores[-1] = reflected_score
                sorted_vertices[-1] = reflected_vertex
                print("score(x_e) >= score(x_r)", expanded_score, " >= ", reflected_score)
                # back to first step: order

        elif reflected_score >= worst:
            print("CONTRACTION")

            contracted_vertex = centroid + rho * (reflected_vertex - centroid)
            denorm_contracted = denormalize_vertex(contracted_vertex, ranges)
            contracted_score = vertex_simulation(denorm_contracted, simul_index, masks)
            simul_index += 1
            print("score(x_r) < score(x_(n+1))", reflected_score, " >= ", worst)

            if contracted_score < worst: 
                simplex_scores[-1] = contracted_score
                sorted_vertices[-1] = contracted_vertex
                print("score(x_r) < score(x_(n+1))", contracted_score, " < ", reflected_score)
                # back to first step: order
                
            else: 
                print("SHRINKAGE")
                best_point = sorted_vertices[0]
                best_score = simplex_scores[0]
                sorted_vertices, simplex_scores, simul_index = shrink(best_point, sorted_vertices[1:], sigma, simul_index, ranges, masks)
                sorted_vertices.append(best_point)
                simplex_scores.append(best_score)
                print("Shrinking the simplex")

        else:
            print("LAST BRANCH")
            simplex_scores[-1] = reflected_score
            sorted_vertices[-1] = reflected_vertex
            print("score(x_r) >= worst:", reflected_score, ">= ", worst)
       
        simplex = sorted_vertices

        print(simplex_scores)
        sys.stdout.flush()

    print(simul_index)



if __name__ == "__main__": 
    easting_range = [vent_location[0] - location_error, vent_location[0] + location_error] 
    northing_range = [vent_location[1] - location_error, vent_location[1] + location_error] 
    ranges = [easting_range, northing_range, h2o_range, flux_pct_range]

    nelder_mead(ranges)

