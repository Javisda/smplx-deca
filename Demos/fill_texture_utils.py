import numpy as np

def viewArea(texture, x_low_limit, x_high_limit, y_low_limit, y_high_limit):
    pixel_x_begin = x_low_limit
    pixel_x_end = x_high_limit
    pixel_y_begin = y_low_limit
    pixel_y_end = y_high_limit
    for pixel_x in range(texture.shape[0]):
        for pixel_y in range(texture.shape[1]):
            if(pixel_y >= pixel_x_begin and pixel_y <=pixel_x_end) and \
                (pixel_x >= pixel_y_begin and pixel_x <= pixel_y_end):
                texture[pixel_x, pixel_y, 2] = 255 # BGR
    return texture


def repairArea(texture, x_low_limit, x_high_limit, y_low_limit, y_high_limit):
    # From left to right in the highlighted area. That is the generated in viewArea
    # and is focused in the forehead area, that is mostly black

    pixel_x_begin = x_low_limit
    pixel_x_end = x_high_limit
    pixel_y_begin = y_low_limit
    pixel_y_end = y_high_limit

    new_texture = texture.copy()

    # Calcular el área de luminancia
    area_luminance_factor = 0
    for x in range(pixel_x_begin, pixel_x_end):
        for y in range(pixel_y_begin, pixel_y_end):
            area_luminance_factor += 0.2126 * texture[y, x, 2] + 0.7152 * texture[y, x, 1]  + 0.0722 * texture[y, x, 0]
    area_luminance_factor = (area_luminance_factor/((pixel_x_end-pixel_x_begin)*(pixel_y_end-pixel_y_begin)))

    for x in range(pixel_x_begin, pixel_x_end):
        for y in range(pixel_y_begin, pixel_y_end):
            pixel_luminance_factor = 0.2126 * texture[y, x, 2] + 0.7152 * texture[y, x, 1]  + 0.0722 * texture[y, x, 0]

            if (pixel_luminance_factor < area_luminance_factor):
                new_texture[y, x, :] = find_closest_coloured_pixels(texture, x, y, y_low_limit, y_high_limit, area_luminance_factor)
    return new_texture



def find_closest_coloured_pixels(texture, current_x, current_y, low_limit_y, high_limit_y, area_luminance_factor):

    # Initial pixel color
    initial_pixel_color = texture[current_y, current_x, :]

    # Boundaries
    low_limit_y = low_limit_y
    high_limit_y = high_limit_y

    # Indexes
    low_y_index = current_y
    high_y_index = current_y


    # Find y top index
    for y in range(current_y, low_limit_y, -1):
        pixel_luminance_factor = 0.2126 * texture[y, current_x, 2] + 0.7152 * texture[y, current_x, 1] + 0.0722 * texture[y, current_x, 0]
        if(pixel_luminance_factor > area_luminance_factor):
            low_y_index = y - 1
            break

    for y in range(current_y, high_limit_y, 1):
        pixel_luminance_factor = 0.2126 * texture[y, current_x, 2] + 0.7152 * texture[y, current_x, 1] + 0.0722 * texture[y, current_x, 0]
        if(pixel_luminance_factor > area_luminance_factor):
            high_y_index = y + 1
            break

    # Once we have the nearest non-black pixels in y coordinates, desired color can be computed.
    # First calculate the factors
    y_top_to_bottom_distance = high_y_index - low_y_index

    if y_top_to_bottom_distance == 0:
        y_top_to_bottom_distance = 1

    y_top_factor = (current_y - low_y_index) / (y_top_to_bottom_distance)
    y_bottom_factor = (high_y_index - current_y) / (y_top_to_bottom_distance)

    color = ((texture[low_y_index, current_x, :] * y_bottom_factor) +
             (texture[high_y_index, current_x, :] * y_top_factor))

    color = color.astype(int)

    # Check if black area
    if ((low_y_index == current_y) and (high_y_index == current_y)):
        color = initial_pixel_color

    return color



def fill_border(texture,x_min, x_max, y_min, y_max,  mode=None):

    if mode is "left_to_right":
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                new_color = (texture[y, x - 1, :] * 0.5) + (texture[y, x - 2, :] * 0.5)
                texture[y, x, :] = new_color.astype(np.int)
    elif mode is "right_to_left":
        for x in range(x_max, x_min - 1, -1):
            for y in range(y_min, y_max + 1):
                new_color = (texture[y, x + 1, :] * 0.5) + (texture[y, x + 2, :] * 0.5)
                texture[y, x, :] = new_color.astype(np.int)
    elif mode is "top_to_bottom":
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                new_color = (texture[y - 1, x, :] * 0.5) + (texture[y - 2, x, :] * 0.5)
                texture[y, x, :] = new_color.astype(np.int)
    elif mode is "bottom_to_top": # Pensado específicamente para la zona de la coronilla
        for x in range(x_min, x_max + 1):
            for y in range(y_max, y_min - 1, - 1):
                new_color = (texture[y + 1, x, :] * 0.5) + (texture[y + 2, x, :] * 0.5)
                # ----
                if((y == y_max)): # Pequeño suavizado en la coronilla
                    symetry_color = (texture[y + 1, -x, :] * 0.5) + (texture[y + 2, -x, :] * 0.5)
                    new_color = (new_color * 0.7) + (symetry_color * 0.3)
                # ----
                texture[y, x, :] = new_color.astype(np.int)
    else:
        print("Incorrect mode.")

    return texture

def select_face_borders(texture):
    # Zonas grandes
    #texture = viewArea(texture, x_low_limit=0, x_high_limit=2, y_low_limit=78, y_high_limit=255, color=[255, 0, 0]) # izquierda
    #texture = viewArea(texture, x_low_limit=3, x_high_limit=251, y_low_limit=253, y_high_limit=255, color=[0, 255, 0]) # debajo
    #texture = viewArea(texture, x_low_limit=252, x_high_limit=255, y_low_limit=78, y_high_limit=255, color=[0, 0, 255]) # derecha

    texture = fill_border(texture, 0, 2, 78, 255, mode="right_to_left")
    texture = fill_border(texture, 3, 251, 253, 255, mode="top_to_bottom")
    texture = fill_border(texture, 252, 255, 78, 255, mode="left_to_right")

    # Color smoothing in the back
    for y in range(78, 255 + 1):
        for x in range(0, 10):
            color = (texture[y, x, :] * 0.5) + (texture[y, -x, :] * 0.5)
            color = color.astype(np.int)
            texture[y, x, :] = color
            texture[y, -x, :] = color



    # Zonas pequeñas
    # texture = view_crown_area(texture)
    texture = repair_crown_area(texture)

    return texture

def view_crown_area(texture):
    texture = viewArea(texture, x_low_limit=122, x_high_limit=131, y_low_limit=18, y_high_limit=20, color=[255, 0, 0])

    texture = viewArea(texture, x_low_limit=111, x_high_limit=122, y_low_limit=16, y_high_limit=18, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=132, x_high_limit=143, y_low_limit=16, y_high_limit=18, color=[255, 0, 0])

    texture = viewArea(texture, x_low_limit=99, x_high_limit=110, y_low_limit=14, y_high_limit=16, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=144, x_high_limit=154, y_low_limit=14, y_high_limit=16, color=[255, 0, 0])

    texture = viewArea(texture, x_low_limit=69, x_high_limit=100, y_low_limit=12, y_high_limit=14, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=155, x_high_limit=186, y_low_limit=12, y_high_limit=14, color=[255, 0, 0])

    texture = viewArea(texture, x_low_limit=53, x_high_limit=68, y_low_limit=14, y_high_limit=18, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=187, x_high_limit=201, y_low_limit=14, y_high_limit=18, color=[255, 0, 0])

    texture = viewArea(texture, x_low_limit=42, x_high_limit=52, y_low_limit=18, y_high_limit=25, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=202, x_high_limit=213, y_low_limit=18, y_high_limit=25, color=[255, 0, 0])

    texture = viewArea(texture, x_low_limit=38, x_high_limit=41, y_low_limit=24, y_high_limit=26, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=214, x_high_limit=217, y_low_limit=24, y_high_limit=26, color=[255, 0, 0])

    texture = viewArea(texture, x_low_limit=36, x_high_limit=38, y_low_limit=26, y_high_limit=28, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=217, x_high_limit=219, y_low_limit=26, y_high_limit=28, color=[255, 0, 0])

    # ----- Zona a partir de los ojos izquierda
    texture = viewArea(texture, x_low_limit=35, x_high_limit=35, y_low_limit=28, y_high_limit=30, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=34, x_high_limit=34, y_low_limit=29, y_high_limit=31, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=33, x_high_limit=33, y_low_limit=30, y_high_limit=32, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=32, x_high_limit=32, y_low_limit=31, y_high_limit=33, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=31, x_high_limit=31, y_low_limit=32, y_high_limit=34, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=30, x_high_limit=30, y_low_limit=33, y_high_limit=35, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=29, x_high_limit=29, y_low_limit=34, y_high_limit=36, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=28, x_high_limit=28, y_low_limit=35, y_high_limit=37, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=27, x_high_limit=27, y_low_limit=36, y_high_limit=38, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=26, x_high_limit=26, y_low_limit=37, y_high_limit=39, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=25, x_high_limit=25, y_low_limit=38, y_high_limit=40, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=23, x_high_limit=24, y_low_limit=39, y_high_limit=42, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=20, x_high_limit=22, y_low_limit=42, y_high_limit=46, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=18, x_high_limit=20, y_low_limit=46, y_high_limit=50, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=16, x_high_limit=18, y_low_limit=50, y_high_limit=52, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=14, x_high_limit=16, y_low_limit=53, y_high_limit=56, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=12, x_high_limit=14, y_low_limit=57, y_high_limit=60, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=10, x_high_limit=12, y_low_limit=61, y_high_limit=65, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=6, x_high_limit=10, y_low_limit=65, y_high_limit=68, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=4, x_high_limit=6, y_low_limit=69, y_high_limit=72, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=0, x_high_limit=4, y_low_limit=73, y_high_limit=77, color=[255, 0, 0])
    # ----- Zona a partir de los ojos derecha
    texture = viewArea(texture, x_low_limit=220, x_high_limit=220, y_low_limit=28, y_high_limit=30, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=221, x_high_limit=221, y_low_limit=29, y_high_limit=31, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=222, x_high_limit=222, y_low_limit=30, y_high_limit=32, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=223, x_high_limit=223, y_low_limit=31, y_high_limit=33, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=224, x_high_limit=224, y_low_limit=32, y_high_limit=34, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=225, x_high_limit=225, y_low_limit=33, y_high_limit=35, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=226, x_high_limit=226, y_low_limit=34, y_high_limit=36, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=227, x_high_limit=227, y_low_limit=35, y_high_limit=37, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=228, x_high_limit=229, y_low_limit=36, y_high_limit=38, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=230, x_high_limit=231, y_low_limit=37, y_high_limit=40, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=232, x_high_limit=233, y_low_limit=38, y_high_limit=44, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=234, x_high_limit=236, y_low_limit=44, y_high_limit=48, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=236, x_high_limit=238, y_low_limit=46, y_high_limit=52, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=238, x_high_limit=242, y_low_limit=47, y_high_limit=58, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=242, x_high_limit=245, y_low_limit=56, y_high_limit=66, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=246, x_high_limit=249, y_low_limit=64, y_high_limit=70, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=250, x_high_limit=251, y_low_limit=68, y_high_limit=75, color=[255, 0, 0])
    texture = viewArea(texture, x_low_limit=252, x_high_limit=255, y_low_limit=72, y_high_limit=77, color=[255, 0, 0])

    return texture

def repair_crown_area(texture):
    texture = fill_border(texture, 122, 131, 18, 20, mode="bottom_to_top")

    texture = fill_border(texture, 111, 122, 16, 18, mode="bottom_to_top")
    texture = fill_border(texture, 132, 143, 16, 18, mode="bottom_to_top")

    texture = fill_border(texture, 99, 110, 14, 16, mode="bottom_to_top")
    texture = fill_border(texture, 144, 154, 14, 16, mode="bottom_to_top")

    texture = fill_border(texture, 69, 100, 12, 14, mode="bottom_to_top")
    texture = fill_border(texture, 155, 186, 12, 14, mode="bottom_to_top")

    texture = fill_border(texture, 53, 68, 14, 18, mode="bottom_to_top")
    texture = fill_border(texture, 187, 201, 14, 18, mode="bottom_to_top")

    texture = fill_border(texture, 42, 52, 18, 25, mode="bottom_to_top")
    texture = fill_border(texture, 202, 213, 18, 25, mode="bottom_to_top")

    texture = fill_border(texture, 38, 41, 24, 26, mode="bottom_to_top")
    texture = fill_border(texture, 214, 217, 24, 26, mode="bottom_to_top")

    texture = fill_border(texture, 36, 38, 26, 28, mode="bottom_to_top")
    texture = fill_border(texture, 217, 219, 26, 28, mode="bottom_to_top")

    # ----- Zona a partir de los ojos izquierda
    texture = fill_border(texture, 35, 35, 26, 30, mode="bottom_to_top")
    texture = fill_border(texture, 34, 34, 27, 31, mode="bottom_to_top")
    texture = fill_border(texture, 33, 33, 28, 32, mode="bottom_to_top")
    texture = fill_border(texture, 32, 32, 29, 33, mode="bottom_to_top")
    texture = fill_border(texture, 31, 31, 30, 34, mode="bottom_to_top")
    texture = fill_border(texture, 30, 30, 31, 35, mode="bottom_to_top")
    texture = fill_border(texture, 29, 29, 32, 36, mode="bottom_to_top")
    texture = fill_border(texture, 28, 28, 33, 37, mode="bottom_to_top")
    texture = fill_border(texture, 27, 27, 34, 38, mode="bottom_to_top")
    texture = fill_border(texture, 26, 26, 35, 39, mode="bottom_to_top")
    texture = fill_border(texture, 25, 25, 36, 40, mode="bottom_to_top")
    texture = fill_border(texture, 23, 24, 37, 42, mode="bottom_to_top")
    texture = fill_border(texture, 20, 22, 40, 46, mode="bottom_to_top")
    texture = fill_border(texture, 18, 20, 44, 50, mode="bottom_to_top")
    texture = fill_border(texture, 16, 18, 48, 52, mode="bottom_to_top")
    texture = fill_border(texture, 14, 16, 51, 56, mode="bottom_to_top")
    texture = fill_border(texture, 12, 14, 55, 60, mode="bottom_to_top")
    texture = fill_border(texture, 10, 12, 59, 65, mode="bottom_to_top")
    texture = fill_border(texture, 6, 10, 63, 68, mode="bottom_to_top")
    texture = fill_border(texture, 4, 6, 67, 72, mode="bottom_to_top")
    texture = fill_border(texture, 0, 4, 71, 77, mode="bottom_to_top")

    # ----- Zona a partir de los ojos derecha
    texture = fill_border(texture, 220, 220, 27, 30, mode="bottom_to_top")
    texture = fill_border(texture, 221, 221, 29, 31, mode="bottom_to_top")
    texture = fill_border(texture, 222, 222, 30, 32, mode="bottom_to_top")
    texture = fill_border(texture, 223, 223, 31, 33, mode="bottom_to_top")
    texture = fill_border(texture, 224, 224, 32, 34, mode="bottom_to_top")
    texture = fill_border(texture, 225, 225, 33, 35, mode="bottom_to_top")
    texture = fill_border(texture, 226, 226, 34, 36, mode="bottom_to_top")
    texture = fill_border(texture, 227, 227, 35, 37, mode="bottom_to_top")
    texture = fill_border(texture, 228, 229, 36, 38, mode="bottom_to_top")
    texture = fill_border(texture, 230, 231, 37, 40, mode="bottom_to_top")
    texture = fill_border(texture, 232, 233, 38, 44, mode="bottom_to_top")
    texture = fill_border(texture, 234, 236, 44, 48, mode="bottom_to_top")
    texture = fill_border(texture, 236, 238, 46, 52, mode="bottom_to_top")
    texture = fill_border(texture, 238, 242, 47, 58, mode="bottom_to_top")
    texture = fill_border(texture, 242, 245, 56, 66, mode="bottom_to_top")
    texture = fill_border(texture, 246, 249, 64, 70, mode="bottom_to_top")
    texture = fill_border(texture, 250, 251, 68, 75, mode="bottom_to_top")
    texture = fill_border(texture, 252, 255, 72, 77, mode="bottom_to_top")

    return texture