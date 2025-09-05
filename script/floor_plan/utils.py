import numpy as np
import math, cv2
import copy
from scipy import ndimage as ndi
from skimage.morphology._skeletonize_cy import (_skeletonize_loop,_table_lookup_index)

_eight_connect = ndi.generate_binary_structure(2, 1)

def prepareRTObj(map):
  """占位函数，保持兼容性"""
  gridsize = (map.shape[0],map.shape[1],0)
  return None

def laserscan_sim(map, point, range=9999.0, resolution=1.0, fov=(0,360)):
  """使用纯Python实现的激光雷达模拟"""
  ranges = []
  
  for deg in np.arange(fov[0],fov[1],resolution):
    # 计算射线方向
    rad = np.deg2rad(deg)
    dx = np.cos(rad)
    dy = np.sin(rad)
    
    # 射线追踪
    x, y = point[0], point[1]
    step = 0.1
    distance = 0
    
    while distance < range:
      x += dx * step
      y += dy * step
      distance += step
      
      # 检查边界
      if x < 0 or x >= map.shape[1] or y < 0 or y >= map.shape[0]:
        break
      
      # 检查碰撞
      if not map[int(y), int(x)]:
        ranges.append(distance)
        break
    else:
      # 如果没有碰撞，使用最大距离
      ranges.append(range)

  return ranges

# def lerp2D(mat, p):
  
#   l=p.astype(int)
#   u=l+1
#   f=p-l
#   m=1.0-f

#   a=mat[l[0], l[1]]*m[1] + mat[l[0], u[1]]*f[1] 
#   b=mat[u[0], l[1]]*m[1] + mat[u[0], u[1]]*f[1] 
#   c=a*m[0] + b*f[0] 

#   return c

def lerp2D(mat, p):
  
  l=p.astype(int)
  f=p-l

  return (mat[l[0], l[1]]*(1.0-f[1]) + mat[l[0], l[1]+1]*f[1])*(1.0-f[0]) + (mat[l[0]+1, l[1]]*(1.0-f[1]) + mat[l[0]+1, l[1]+1]*f[1]) *f[0] 

def laserscan_sim_raymarching(dist, point, range=9999.0, resolution=1.0, fov=(0,360)):

  angs = np.arange(np.deg2rad(fov[0]),np.deg2rad(fov[1]),np.deg2rad(resolution))
  rays = np.stack((np.sin(angs), np.cos(angs)), axis=-1)
  ranges = np.zeros(angs.shape[0], dtype=float)
  origin = np.asarray(point, dtype=float)

  i=-1
  for ray in rays:
    i+=1
    p = origin
    while True:
      d = lerp2D(dist, p)
      p = p + ray * d
      ranges[i] += d
      if d<0.1:
        break

  return ranges

def laserscan_sim_raymarching_fast(dist, point, range=9999.0, resolution=1.0, fov=(0,360)):

  angs = np.arange(np.deg2rad(fov[0]),np.deg2rad(fov[1]),np.deg2rad(resolution))
  rays = np.stack((np.sin(angs), np.cos(angs)), axis=-1)
  ranges = np.zeros(angs.shape[0], dtype=float)
  origin = np.asarray(point, dtype=float)

  i=0
  for ray in rays:

    p = origin.copy()
    ret = p.astype(int)
    
    while True:

      # raymarching
      ranges[i] += dist[ret[0], ret[1]]
      _ret = (origin + ranges[i]*ray).astype(int)

      # if ray stop
      if ret[0]==_ret[0] and ret[1]==_ret[1]:

        # sub-pixel searching
        while dist[ret[0], ret[1]]>0:
          ranges[i] += 1.5
          ret = (origin + ranges[i]*ray).astype(int)
            
        while dist[ret[0], ret[1]]==0:
          ranges[i] -= 0.1
          ret = (origin + ranges[i]*ray).astype(int)

        break

      ret=_ret

    i+=1

  return ranges

def getRayVector(angle, length=1, xOffset=0, yOffset=0):
  angle = angle * math.pi / 180; 
  return (length * math.sin(angle) + xOffset, length * math.cos(angle) + yOffset)

# def extract_edges(vert_map, map):
#   connections = []
#   vertex = vert_map.nonzero()
#   visited = np.full((map.shape[0], map.shape[1]),False,dtype='bool')
#   _four_connect = np.array([[False, True, False], [True, False, True], [False, True, False]])
#   for i,j in zip(vertex[0], vertex[1]): #for each v, follow its neighbour with pruned medial edge
#     #find all neighbour
#     nb = (map[i-1:i+2, j-1:j+2]*_four_connect).nonzero()
#     visited[i,j] = True

#     for u,v in zip(nb[0], nb[1]):
#       path_length = 1
#       x=i+(u-1) #init
#       y=j+(v-1) 

#       while(True):
#         if vert_map[x,y]:
#           # print('connection found')
#           # print('origin: ', i, j, 'dest: ',x, y)
#           connections.append(([i,j],[x,y],path_length))
#           break
#         else:
#           nx,ny = (map[x-1:x+2,y-1:y+2]*_four_connect*~visited[x-1:x+2,y-1:y+2]).nonzero()
#           if not len(nx):
#             break
#           visited[x,y] = True
#           x = x+(nx[0]-1)
#           y = y+(ny[0]-1)
#           path_length = path_length + 1

#   return connections 

def extract_edges(vert_map, map):
  connections = []
  vertex = vert_map.nonzero()
  visited = np.full((map.shape[0], map.shape[1]),False,dtype='bool')
  _four_connect = np.array([[False, True, False], [True, False, True], [False, True, False]])
  for i,j in zip(vertex[0], vertex[1]): #for each v, follow its neighbour with pruned medial edge
    #find all neighbour
    nb = (~visited[i-1:i+2, j-1:j+2]*map[i-1:i+2, j-1:j+2]*_four_connect).nonzero()
    visited[i,j] = True

    for u,v in zip(nb[0], nb[1]):
      path_length = 1
      x=i+(u-1) #init
      y=j+(v-1) 

      while(True):
        if vert_map[x,y]:
          connections.append(([i,j],[x,y],path_length))
          break
        else:
          nx,ny = (map[x-1:x+2,y-1:y+2]*_four_connect*~visited[x-1:x+2,y-1:y+2]).nonzero()
          if nx.shape[0]==0:
            break
          visited[x,y] = True
          x = x+(nx[0]-1)
          y = y+(ny[0]-1)
          path_length = path_length + 1

  return connections 

def prune_end_point(medial, dist_mask, max_iter=100):
  iter = 0
  med = copy.deepcopy(medial)
  while iter < max_iter:
    ep_map = find_endpoints(med, dist_mask)
    if (np.sum(ep_map) == 0):
      break
    med = np.logical_xor(med, ep_map)
    iter+=1
  return med

# def find_vertices(map, mask=None, end_point_only=False):
#   _four_connect = np.array([[False, True, False], [True, True, True], [False, True, False]], dtype='bool')
#   output_map = np.full((map.shape[0], map.shape[1]),False,dtype='bool')
#   map_padded = np.pad(map,1)
#   if mask is None:
#     mask_padded = np.full((map_padded.shape[0], map_padded.shape[1]),True,dtype='bool')
#   else:
#     mask_padded = np.pad(mask,1)

#   for i in range(map_padded.shape[0]):
#     for j in range(map_padded.shape[1]):
#       if (map_padded*mask_padded)[i,j]: 
#         # vertices: 1 or more than 3 neighbour edge
#         # end points: exactly 1 neighbour edge
#         edgecount = np.sum((map_padded[i-1:i+2,j-1:j+2]*_four_connect).flatten()) #3x3 area around i,j
#         if edgecount==2:
#           output_map[i-1,j-1] = True
#         if (not end_point_only) and edgecount >=4:
#           output_map[i-1,j-1] = True

#   return output_map

def find_vertices(map):
  _four_connect = np.array([[False, True, False], [True, False, True], [False, True, False]], dtype='bool')
  output_map = np.full((map.shape[0], map.shape[1]),False,dtype='bool')
  map_padded = np.pad(map,1)
  mask_padded = np.full((map_padded.shape[0], map_padded.shape[1]),True,dtype='bool')

  # first find all vert coord with numpy
  grid = mask_padded.nonzero()

  # iterate through all the coord
  for (i,j) in zip(grid[0], grid[1]):

    # remove whole matrix multiplication
    if map_padded[i,j]*mask_padded[i,j]: 
      edgecount = np.sum((map_padded[i-1:i+2,j-1:j+2]*_four_connect)) #3x3 area around i,j
      output_map[i-1,j-1] = edgecount>=3
      # output_map[i-1,j-1] = edgecount==1 or edgecount>=3

  return output_map

def find_endpoints(map, mask):
  _four_connect = np.array([[False, True, False], [True, False, True], [False, True, False]], dtype='bool')
  output_map = np.full((map.shape[0], map.shape[1]),False,dtype='bool')
  map_padded = np.pad(map,1)
  mask_padded = np.pad(mask,1)

  # first find all vert coord with numpy
  grid = mask_padded.nonzero()

  # iterate through all the coord
  for (i,j) in zip(grid[0], grid[1]):

    # remove whole matrix multiplication
    if map_padded[i,j]*mask_padded[i,j]: 
      edgecount = np.sum((map_padded[i-1:i+2,j-1:j+2]*_four_connect)) #3x3 area around i,j
      output_map[i-1,j-1] = edgecount==1

  return output_map

def medial_axis_four(image, mask=None, return_distance=False, *, random_state=None):
    """Compute the medial axis transform of a binary image.
    Parameters
    ----------
    image : binary ndarray, shape (M, N)
        The image of the shape to be skeletonized.
    mask : binary ndarray, shape (M, N), optional
        If a mask is given, only those elements in `image` with a true
        value in `mask` are used for computing the medial axis.
    return_distance : bool, optional
        If true, the distance transform is returned as well as the skeleton.
    random_state : {None, int, `numpy.random.Generator`}, optional
        If `random_state` is None the `numpy.random.Generator` singleton is
        used.
        If `random_state` is an int, a new ``Generator`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``Generator`` instance then that
        instance is used.
        .. versionadded:: 0.19
    Returns
    -------
    out : ndarray of bools
        Medial axis transform of the image
    dist : ndarray of ints, optional
        Distance transform of the image (only returned if `return_distance`
        is True)
    See Also
    --------
    skeletonize
    Notes
    -----
    This algorithm computes the medial axis transform of an image
    as the ridges of its distance transform.
    The different steps of the algorithm are as follows
     * A lookup table is used, that assigns 0 or 1 to each configuration of
       the 3x3 binary square, whether the central pixel should be removed
       or kept. We want a point to be removed if it has more than one neighbor
       and if removing it does not change the number of connected components.
     * The distance transform to the background is computed, as well as
       the cornerness of the pixel.
     * The foreground (value of 1) points are ordered by
       the distance transform, then the cornerness.
     * A cython function is called to reduce the image to its skeleton. It
       processes pixels in the order determined at the previous step, and
       removes or maintains a pixel according to the lookup table. Because
       of the ordering, it is possible to process all pixels in only one
       pass.
    Examples
    --------
    >>> square = np.zeros((7, 7), dtype=np.uint8)
    >>> square[1:-1, 2:-2] = 1
    >>> square
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 1, 1, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    >>> medial_axis(square).astype(np.uint8)
    array([[0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0],
           [0, 0, 1, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0]], dtype=uint8)
    """
    global _eight_connect
    if mask is None:
        masked_image = image.astype(bool)
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    #
    # Build lookup table - three conditions
    # 1. Keep only positive pixels (center_is_foreground array).
    # AND
    # 2. Keep if removing the pixel results in a different connectivity
    # (if the number of connected components is different with and
    # without the central pixel)
    # OR
    # 3. Keep if # pixels in neighborhood is 2 or less
    # Note that table is independent of image
    center_is_foreground = (np.arange(512) & 2**4).astype(bool)
    table = (center_is_foreground  # condition 1.
                &
            (np.array([ndi.label(_pattern_of(index), _eight_connect)[1] !=
                       ndi.label(_pattern_of(index & ~ 2**4),
                                    _eight_connect)[1]
                       for index in range(512)])  # condition 2
                |
        np.array([np.sum(_pattern_of(index)) < 3 for index in range(512)]))
        # condition 3
            )

    # Build distance transform
    distance = ndi.distance_transform_edt(masked_image)
    if return_distance:
        store_distance = distance.copy()

    # Corners
    # The processing order along the edge is critical to the shape of the
    # resulting skeleton: if you process a corner first, that corner will
    # be eroded and the skeleton will miss the arm from that corner. Pixels
    # with fewer neighbors are more "cornery" and should be processed last.
    # We use a cornerness_table lookup table where the score of a
    # configuration is the number of background (0-value) pixels in the
    # 3x3 neighborhood
    cornerness_table = np.array([9 - np.sum(_pattern_of(index))
                                 for index in range(512)])
    corner_score = _table_lookup(masked_image, cornerness_table)

    # Define arrays for inner loop
    i, j = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    result = masked_image.copy()
    distance = distance[result]
    i = np.ascontiguousarray(i[result], dtype=np.intp)
    j = np.ascontiguousarray(j[result], dtype=np.intp)
    result = np.ascontiguousarray(result, np.uint8)

    # Determine the order in which pixels are processed.
    # We use a random # for tiebreaking. Assign each pixel in the image a
    # predictable, random # so that masking doesn't affect arbitrary choices
    # of skeletons
    #
    generator = np.random.default_rng(random_state)
    tiebreaker = generator.permutation(np.arange(masked_image.sum()))
    order = np.lexsort((tiebreaker,
                        corner_score[masked_image],
                        distance))
    order = np.ascontiguousarray(order, dtype=np.int32)

    table = np.ascontiguousarray(table, dtype=np.uint8)
    # Remove pixels not belonging to the medial axis
    _skeletonize_loop(result, i, j, order, table)

    result = result.astype(bool)
    if mask is not None:
        result[~mask] = image[~mask]
    if return_distance:
        return result, store_distance
    else:
        return result


def _pattern_of(index):
    """
    Return the pattern represented by an index value
    Byte decomposition of index
    """
    return np.array([[index & 2**0, index & 2**1, index & 2**2],
                     [index & 2**3, index & 2**4, index & 2**5],
                     [index & 2**6, index & 2**7, index & 2**8]], bool)


def _table_lookup(image, table):
    """
    Perform a morphological transform on an image, directed by its
    neighbors
    Parameters
    ----------
    image : ndarray
        A binary image
    table : ndarray
        A 512-element table giving the transform of each pixel given
        the values of that pixel and its 8-connected neighbors.
    Returns
    -------
    result : ndarray of same shape as `image`
        Transformed image
    Notes
    -----
    The pixels are numbered like this::
      0 1 2
      3 4 5
      6 7 8
    The index at a pixel is the sum of 2**<pixel-number> for pixels
    that evaluate to true.
    """
    #
    # We accumulate into the indexer to get the index into the table
    # at each point in the image
    #
    if image.shape[0] < 3 or image.shape[1] < 3:
        image = image.astype(bool)
        indexer = np.zeros(image.shape, int)
        indexer[1:, 1:]   += image[:-1, :-1] * 2**0
        indexer[1:, :]    += image[:-1, :] * 2**1
        indexer[1:, :-1]  += image[:-1, 1:] * 2**2

        indexer[:, 1:]    += image[:, :-1] * 2**3
        indexer[:, :]     += image[:, :] * 2**4
        indexer[:, :-1]   += image[:, 1:] * 2**5

        indexer[:-1, 1:]  += image[1:, :-1] * 2**6
        indexer[:-1, :]   += image[1:, :] * 2**7
        indexer[:-1, :-1] += image[1:, 1:] * 2**8
    else:
        indexer = _table_lookup_index(np.ascontiguousarray(image, np.uint8))
    image = table[indexer]
    return image