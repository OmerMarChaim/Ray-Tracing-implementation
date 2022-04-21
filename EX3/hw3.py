from numpy import intersect1d
from helper_classes import *
import matplotlib.pyplot as plt

# EPSILON = 1e-5
EPSILON = 0.00001


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
            intersection_point = scene_ray.get_intersection_point(min_distance )
            # According to the last tip - shift the point in the direction on epsilon normal
            normal_to_surface = normalize(nearest_object.get_normal())
            if isinstance(nearest_object, Sphere):
                normal_to_surface = normalize(intersection_point - nearest_object.center)
            shifted_point = intersection_point + EPSILON * normal_to_surface
            direction_to_eye = normalize(camera - shifted_point)

            color = get_color(scene_ray, ambient, lights, objects, 1, max_depth, direction_to_eye, camera,
                              normal_to_surface, shifted_point,nearest_object)

            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color, 0, 1)

    return image


# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0, 0, 1])
    lights = []
    objects = []
    sphere_a = Sphere([-0.5, 0.2, -1],0.5)
    sphere_a.set_material([1, 0, 0], [1, 0, 0], [0.3, 0.3, 0.3], 100, 1)
    background = Plane([0,0,1], [0,0,-3])
    background.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], 1000, 0.5)
    triangle = Triangle([1,-1,-2],[0,1,-1.5],[0,-1,-1])
    triangle.set_material([1, 1, 0], [1, 1, 0], [0, 1, 0], 100, 0.5)
    
    plane = Plane([0,0,1], [0,0,-3])
    plane.set_material([0, 0.5, 0], [0, 1, 1], [1, 1, 1], 100, 0.5)
    objects = [sphere_a,triangle,plane,background]
    light_a = PointLight(intensity= np.array([1, 1, 1]),position=np.array([1,1.5,1]),kc=0.1,kl=0.1,kq=0.1)
    light_b = SpotLight(intensity= np.array([1, 0, 0]),position=np.array([0,-0.5,0]), direction=([0,0,1]),
                        kc=0.1,kl=0.1,kq=0.1)

    lights = [light_a,light_b]
    ambient = np.array([0.1,0.2,0.3])

    camera = np.array([0,0,1])
    return camera, lights, objects, ambient

# def get_diffuse(nearest_object,)
