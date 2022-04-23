from numpy import intersect1d
from helper_classes import *
import matplotlib.pyplot as plt

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
            nearest_object, min_distance = scene_ray.nearest_intersected_object(objects)
            if nearest_object is None:
                continue
            intersection_point = scene_ray.get_intersection_point(min_distance - EPSILON)
            # According to the last tip - shift the point in the direction on epsilon normal
            normal_to_surface = normalize(nearest_object.get_normal())
            if isinstance(nearest_object, Sphere):
                normal_to_surface = normalize(intersection_point - nearest_object.center)

            color = get_color(scene_ray, ambient, lights, objects, 1, max_depth, normal_to_surface, intersection_point,
                              nearest_object)

            image[i, j] = np.clip(color, 0, 1)

    return image


# Write your own objects and lights
def your_own_scene():
    camera = np.array([0, 0, 1])
    lights = []
    objects = []
    sphere_a = Sphere([-0.5, 0.2, -1], 0.5)
    sphere_a.set_material([1, 0, 1], [1, 0, 1], [0.3, 0.3, 0.3], 100, 1)
    background = Plane([0, 0, 1], [5, 5, -8])
    background.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], 1000, 0.5)
    v_list = np.array([[-1, -1, -2], [1, -1, -2], [0, -1, -1], [0, 1, -1.5]])
    f_list = np.array([[0, 2, 1], [0, 1, 3], [0, 2, 3], [1, 3, 2]])

    mesh = Mesh(v_list, f_list)
    mesh.set_material([0.3, 0.5, 1], [0.3, 0.5, 0.8], [0.3, 0.3, 0.3], 10, 0.5)
    mesh.apply_materials_to_triangles()
    plane = Plane([0, 0, 1], [0, 0, -3])
    plane.set_material([0, 0.5, 0], [0, 1, 1], [1, 1, 1], 100, 0.5)
    objects = [sphere_a, plane, mesh, background]
    light_a = PointLight(intensity=np.array([1, 1, 1]), position=np.array([1, 1.5, 1]), kc=0.1, kl=0.1, kq=0.1)
    light_b = SpotLight(intensity=np.array([1, 0, 0]), position=np.array([0, -0.5, 0]), direction=([0, 0, 1]),
                        kc=0.1, kl=0.1, kq=0.1)

    lights = [light_a, light_b]
    ambient = np.array([0.1, 0.2, 0.3])

    camera = np.array([0, 0, 1])
    return camera, lights, objects, ambient

# def get_diffuse(nearest_object,)
