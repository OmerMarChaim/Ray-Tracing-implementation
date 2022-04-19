from numpy import intersect1d
from helper_classes import *
import matplotlib.pyplot as plt

EPSILON = 1e-5


def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            pixel = np.array([x, y, 0])
            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            # define the ray with the origin and direction
            origin = normalize(pixel - camera)
            scene_ray = Ray(camera, origin)
            # First search if there is intersection and the object that we intersect
            nearest_object, min_distance = scene_ray.nearest_intersected_object(objects)
            if nearest_object is None:
                continue
            intersection_point = scene_ray.get_intersection_point(min_distance)
            # According to the last tip - shift the point in the direction on epsilon normal
            normal_to_surface = normalize(nearest_object.get_normal())
            shifted_point = intersection_point + EPSILON * normal_to_surface
            direction_to_eye = normalize(camera - shifted_point)
            color = get_color(scene_ray, ambient, lights, objects,
                              1, max_depth, direction_to_eye, camera, normal_to_surface, shifted_point, nearest_object)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0, 0, 1])
    lights = []
    objects = []
    return camera, lights, objects

# def get_diffuse(nearest_object,)
