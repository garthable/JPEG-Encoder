import numpy as np

def run_dct_on_image(image):
    subimages = segment_image(image)
    xlen = (len(image[0]))
    x = 0
    y = 0
    for subimage in subimages:
        chunk = dct(subimage)
        print(len(chunk[0]))
        for j in range(8):
            for k in range(8):
                image[y+j][x+k] = chunk[j][k]
        x += 8
        if x >= xlen:
            x = 0
            y += 8
    return image

def run_inverse_dct_on_image(image):
    return image

def inverse_dct(image):
    return image

# https://dev.to/marycheung021213/understanding-dct-and-quantization-in-jpeg-compression-1col

def segment_image(image):
    h, w = image.shape
    assert h % 8 == 0, f"{h} rows is not evenly divisible by {8}"
    assert w % 8 == 0, f"{w} cols is not evenly divisible by {8}"
    return (image.reshape(h//8, 8, -1, 8)
            .swapaxes(1,2)
            .reshape(-1, 8, 8))

def C(z: float):
    if z == 0:
        return 1.0/np.sqrt(2)
    else:
        return 1

def dctF(imageSeg: np.ndarray[np.ndarray[float]], u: int, v: int):
    sum = 0.0
    for x in range(8):
        for y in range(8):
            a = (np.cos((2*x + 1)*u*np.pi/16))
            b = (np.cos((2*y + 1)*v*np.pi/16))
            sum += imageSeg[x][y]*a*b
    return 1.0/4*C(u)*C(v)*sum

def dct(imageSeg: np.ndarray[np.ndarray[float]]):
    out: np.ndarray[np.ndarray[float]] = np.zeros((8,8))
    for u in range(8):
        for v in range(8):
            out[u][v] = dctF(imageSeg, u, v)
    return out


a = np.array([[144,139,149,155,153,155,155,155],
				[151,151,151,159,156,156,156,158],
				[151,156,160,162,159,151,151,151],
				[158,163,161,160,160,160,160,161],
				[158,160,161,162,160,155,155,156],
				[161,161,161,161,160,157,157,157],
				[162,162,161,160,161,157,157,157],
				[162,162,161,160,163,157,158,154],
                [144,139,149,155,153,155,155,155],
				[151,151,151,159,156,156,156,158],
				[151,156,160,162,159,151,151,151],
				[158,163,161,160,160,160,160,161],
				[158,160,161,162,160,155,155,156],
				[161,161,161,161,160,157,157,157],
				[162,162,161,160,161,157,157,157],
				[162,162,161,160,163,157,158,154]])

print(run_dct_on_image(a))