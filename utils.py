"""
This is the utils module. It contains utility functions for mathematical operations, file handling, and other miscellaneous tasks.
"""

import os
import random
import string
import shutil
import time
from .nodeIntersection import (
    do_meshes_intersect,
    compute_intersection_points,
    panda_mesh_to_numpy,
    Mgr as NodeIntersection,
    Sphere,
    Cube,
    BaseActor,
    BaseCollider,
    ComplexActor,
    ComplexCollider,
    CollisionReport,
    CubeGenerator,
)

# import opensimplex


class Misc:
    @staticmethod
    def get_random_string(length=10):
        """Generates a random string of fixed length."""
        letters = string.ascii_letters
        return "".join(random.choice(letters) for i in range(length))


class Math:
    @staticmethod
    def divideWithRemainder(a, b):
        """Returns the quotient and remainder of a divided by b."""
        return (a // b, a % b)

    @staticmethod
    def random_number(min, max):
        """Returns a random number between min and max."""
        return random.randint(min, max)

    @staticmethod
    def random_float(min, max):
        """Returns a random float between min and max."""
        return random.uniform(min, max)

    @staticmethod
    def random_vector(min, max):
        """Returns a random vector between min and max."""
        return (
            random.uniform(min, max),
            random.uniform(min, max),
            random.uniform(min, max),
        )

    @staticmethod
    def random_color():
        """Returns a random color."""
        return (
            random.random(),
            random.random(),
            random.random(),
        )


class File:
    @staticmethod
    def get_file_name_from_path(path):
        """Returns the file name from a given path."""
        return os.path.basename(path)

    @staticmethod
    def get_file_extension(file_name):
        """Returns the file extension from a given file name."""
        return os.path.splitext(file_name)[1]

    @staticmethod
    def get_file_name_without_extension(file_name):
        """Returns the file name without the extension from a given file name."""
        return os.path.splitext(file_name)[0]

    @staticmethod
    def read_file(file_path):
        """Reads a file and returns its content."""
        with open(file_path, "r") as file:
            return file.read()

    @staticmethod
    def write_file(file_path, content):
        """Writes content to a file."""
        with open(file_path, "w") as file:
            file.write(content)

    @staticmethod
    def delete_file(file_path):
        """Deletes a file."""
        os.remove(file_path)

    @staticmethod
    def copy_file(src, dst):
        """Copies a file from src to dst."""
        shutil.copy(src, dst)

    @staticmethod
    def move_file(src, dst):
        """Moves a file from src to dst."""
        shutil.move(src, dst)

    @staticmethod
    def get_file_size(file_path):
        """Returns the size of a file in bytes."""
        return os.path.getsize(file_path)

    @staticmethod
    def get_file_creation_time(file_path):
        """Returns the creation time of a file."""
        return os.path.getctime(file_path)

    @staticmethod
    def get_file_modification_time(file_path):
        """Returns the modification time of a file."""
        return os.path.getmtime(file_path)

    @staticmethod
    def get_file_access_time(file_path):
        """Returns the access time of a file."""
        return os.path.getatime(file_path)


class Noise:
    # Requires opensimplex

    # def generate_noise(x, y, seed=None):
    #     """Generates noise using the OpenSimplex algorithm."""
    #     if seed is None:
    #         seed = int(time.time() * 1000) % 1000
    #     opensimplex.seed(seed)
    #     return opensimplex.noise2(x, y)

    # def generate_noise_array(x, y, seed=None):
    #     """Generates noise using the OpenSimplex algorithm."""
    #     if seed is None:
    #         seed = int(time.time() * 1000) % 1000
    #     opensimplex.seed(seed)
    #     return opensimplex.noise2array(x, y)
    None
