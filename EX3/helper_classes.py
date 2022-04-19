import numpy as np
from abc import abstractmethod

EPSILON = 1e-5


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


def subsets(numbers):
    if numbers == []:
        return [[]]
    x = subsets(numbers[1:])
    return x + [[numbers[0]] + y for y in x]


# wrapper function
def subsets_of_given_size(numbers, n):
    return [x for x in subsets(numbers) if len(x) == n]


# TODO:
# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, normal):
    v = vector - 2 * (np.dot(vector, normal)) * normal
    # add(u, mult(-2*dot(u, normal), normal));
    return v


# Lights

# !! DO NOT CHANGE HERE !!


class LightSource:

    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        self.direction = direction

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection_point):
        return Ray(intersection_point, self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        return self.intensity


class PointLight(LightSource):

    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the point to the light source
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point I_L
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl * d + self.kq * (d ** 2))


class SpotLight(LightSource):

    def __init__(self, intensity, direction, position, kc, kl, kq):
        super().__init__(intensity)

        self.direction = normalize(direction)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the point to light source
    def get_light_ray(self, intersection):
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        return np.linalg.norm(intersection - self.position)

    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        v = normalize(self.get_light_ray(intersection).direction)
        return (self.intensity * np.dot(v, self.direction)) / (self.kc + self.kl * d + self.kq * (d ** 2))
        pass


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        nearest_object = None
        min_distance = np.inf
        for curr_object in objects:
            distance, _ = curr_object.intersect(self)
            if distance and distance < min_distance:
                nearest_object = curr_object
                min_distance = distance

        return nearest_object, min_distance

    def get_intersection_point(self, distance):
        point = self.origin + distance * self.direction

        return point


# !! DO NOT CHANGE HERE !!

class Object3D:

    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = np.array(ambient)  # Ka
        self.diffuse = np.array(diffuse)  # Kd
        self.specular = np.array(specular)  # Ks
        self.shininess = shininess  # n
        self.reflection = reflection

    @abstractmethod
    def intersect(self, ray):
        pass


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = (np.dot(v, self.normal) / np.dot(self.normal, ray.direction))
        if t > 0:
            return t, self
        else:
            return None, None

    def get_normal(self):
        return self.normal


class Triangle(Object3D):
    # Triangle gets 3 points as arguments
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.get_normal()

    def get_normal(self):
        u = self.b - self.a
        v = self.c - self.a
        normal = np.cross(u, v)
        return np.array(normalize(normal))

    def intersect(self, ray: Ray):
        plane = Plane(self.normal, self.a)
        intersection_distance, _ = plane.intersect(ray)
        if intersection_distance is None:
            return None, None
        intersection_point = ray.origin + intersection_distance * ray.direction
        is_intersect = self.clac_barycentric(intersection_point)

        if is_intersect:
            return intersection_distance, self
        else:
            return None, None

    def clac_barycentric(self, intersection_point):
        areaABC = np.linalg.norm(np.cross((self.b - self.a), (self.c - self.a))) / 2
        PA = self.a - intersection_point
        PB = self.b - intersection_point
        PC = self.c - intersection_point
        alpha = np.linalg.norm(np.cross(PB, PC)) / (2 * areaABC)
        beta = np.linalg.norm(np.cross(PC, PA)) / (2 * areaABC)
        gamma = np.linalg.norm(np.cross(PA, PB)) / (2 * areaABC)

        return np.abs(alpha + beta + gamma - 1) < EPSILON


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius
        self.intersection = np.zeros(3)

    def intersect(self, ray: Ray):
        a = np.linalg.norm(ray.direction) ** 2
        b = 2 * np.dot(ray.direction, ray.origin - self.center)
        c = np.linalg.norm(ray.origin - self.center) ** 2 - self.radius ** 2
        delta = b ** 2 - 4 * a * c
        if delta <= 0:
            return None, None
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)
        if t1 > 0 and t2 > 0:
            return min(t1, t2), self
        return None, None

    def get_normal(self):
        return normalize(self.intersection - self.center)


class Mesh(Object3D):
    # Mesh are defined by a list of vertices, and a list of faces.
    # The faces are triplets of vertices by their index number.
    def __init__(self, v_list, f_list):
        self.v_list = v_list
        self.f_list = f_list
        self.triangle_list = self.create_triangle_list()
        self.normal = np.zeros(3)

    def create_triangle_list(self):
        l = []
        for f in self.f_list:
            l.append(Triangle(self.v_list[f[0]], self.v_list[f[1]], self.v_list[f[2]]))

        return l

    def apply_materials_to_triangles(self):
        for t in self.triangle_list:
            t.set_material(self.ambient, self.diffuse,
                           self.specular, self.shininess, self.reflection)

    # Hint: Intersect returns both distance and nearest object.
    # Keep track of both.
    def intersect(self, ray: Ray):
        # TODO
        min_distance = np.inf
        intersect_object = None
        for tri in self.triangle_list:
            temp_distance, _ = tri.intersect(ray)
            if temp_distance and temp_distance < min_distance:
                min_distance = temp_distance
                intersect_object = tri

        if intersect_object is None:
            return None, None

        self.normal = intersect_object.get_normal()

        return min_distance, intersect_object

    def get_normal(self):
        return self.normal


def get_diffuse_color(nearest_object, normal_to_surface, ray_to_light):
    return nearest_object.diffuse * (np.dot(normal_to_surface, normalize(ray_to_light.direction)))


def get_specular_color(nearest_object, normal_to_surface, ray_to_light):
    reflected_light_vector = reflected(normalize(-ray_to_light.direction), normal_to_surface)
    return nearest_object.specular * (
            np.dot(-ray_to_light.direction, reflected_light_vector) ** nearest_object.shininess)


def get_color(curr_ray, ambient, lights, objects, depth_level, max_depth, normal_to_surface, intersection_point,
              nearest_object):
    color = np.zeros(3)
    # = obj_ambient*global_ambient=
    ambient = nearest_object.ambient * ambient
    sigma_clac = 0
    reflective_rec_calc = 0
    # color += ambient
    normal_to_surface = nearest_object.get_normal()
    if isinstance(nearest_object, Sphere):
        normal_to_surface = normalize(intersection_point - nearest_object.center)
    shifted_point = intersection_point + EPSILON * normal_to_surface
    for light in lights:
        # get the ray from the inter. point to light source
        ray_to_light = light.get_light_ray(intersection_point)
        _, nearest_point_distance = ray_to_light.nearest_intersected_object(objects)
        intersection_to_light_distance = light.get_distance_from_light(intersection_point)
        if nearest_point_distance < intersection_to_light_distance:
            color += np.zeros(3)
        else:
            diffuse = get_diffuse_color(nearest_object, normal_to_surface, ray_to_light)

            specular = get_specular_color(nearest_object, normal_to_surface, ray_to_light)
            # as we do in the formula, ww multiply the diffuse and the specular in the intensity
            color += (diffuse + specular) * light.get_intensity(intersection_point)

    if max_depth < depth_level + 1:
        return color

    reflected_ray_direction = normalize(reflected(curr_ray.direction, normal_to_surface))

    reflected_ray = Ray(intersection_point, reflected_ray_direction)

    # find the nearest object that the reflected_ray hit
    reflected_nearest_object, min_distance_to_nearest = reflected_ray.nearest_intersected_object(objects)

    if reflected_nearest_object is None:
        # here we dont have any intersection
        return color

    intersection_point = reflected_ray.get_intersection_point(min_distance_to_nearest - EPSILON)
    normal_to_surface = normalize(reflected_nearest_object.get_normal())
    # shifted_point = intersection_point + EPSILON * normal_to_surface
    # recreation call for the last term of the formula
    reflective_rec_calc += nearest_object.reflection * get_color(reflected_ray, ambient, lights, objects, depth_level +
                                                                 1, max_depth,
                                                                 normal_to_surface, intersection_point,
                                                                 reflected_nearest_object)

    return color + reflective_rec_calc
