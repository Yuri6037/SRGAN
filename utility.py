import math


def get_max_region_size(w, h):
    val = math.gcd(w, h)
    while val > 64:
        val /= 2
    if val < 8:
        print("WARNING: The maximum region size is less than 8, this may lower the accuracy of this network.")
    return val


def image_decomposition(img, region_size):
    regions = []
    regions_x = int(img.shape[0] / region_size)
    regions_y = int(img.shape[1] / region_size)
    for i in range(0, regions_x):
        for j in range(0, regions_y):
            rx = i * region_size
            ry = j * region_size
            rx1 = rx + region_size
            ry1 = ry + region_size
            region = img[rx:rx1, ry:ry1]
            regions.append(region)
    return regions


def image_recomposition(img, region_size, regions):
    regions_x = int(img.shape[0] / region_size)
    regions_y = int(img.shape[1] / region_size)
    for i in range(0, regions_x):
        for j in range(0, regions_y):
            region = regions[i * regions_y + j]
            rx = i * region_size
            ry = j * region_size
            rx1 = rx + region_size
            ry1 = ry + region_size
            img[rx:rx1, ry:ry1] = region
    return img
