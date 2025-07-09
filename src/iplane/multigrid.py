'''Creates, manipulates, and provides access to multiple grids for differentiation and integration.'''

"""
A grid is an n-dimensional cube of points.
A multigrid is a collection of grids.
A multigrid provides:
    1. Addition of new grids
    2. Access to individual grids
    3. Operations on all grids (e.g., integration, differentiation)
    4. Iteration over points in the grids without duplication.
"""

import itertools
import numpy as np
import pandas as pd  # type: ignore
from typing import List, Optional, Tuple, Any, cast, Union


class Point(object):
    """A point in n-dimensional space."""
    
    def __init__(self, coordinates: List[float]) -> None:
        """
        Initialize a point with given coordinates.
        
        Args:
            coordinates (List[float]): Coordinates of the point in n-dimensional space.
        """
        self.coordinates = coordinates

    def __repr__(self) -> str:
        return f"Point({self.coordinates})"
    

class Axis(object):
    """An axis in n-dimensional space."""
    
    def __init__(self, min_val: float, max_val: float, increment: float,
            center: Optional[float]=None, name: Optional[str]=None) -> None:
        """
        Initialize an axis with a name, minimum and maximum values, and number of points.
        Min and max values are approximate based on the increment.
        
        Args:
            name (str): Name of the axis.
            min_val (float): Minimum value of the axis.
            max_val (float): Maximum value of the axis.
            increment (float): Increment value for the axis.
            center (Optional[float]): Center value of the axis. If None, it is calculated
                as the average of min_val and max_val.
            name: str: Name of the axis.
        """
        if name is None:
            name = f"A{np.random.randint(10000, 99999)}"
        self.name = name
        self.increment = increment
        if center is None:
            center = (min_val + max_val) / 2
        self.center = center
        self.min_val = self.increment*int((self.center - min_val)/self.increment)
        self.max_val = self.increment*int((max_val - self.center)/self.increment)
        self.num_point = int((max_val - min_val) / self.increment) + 1
        # Calculate the positions of the axis relative to its center.
        num_left_coordinate = int(self.center - self.min_val // self.increment)
        num_right_coordinate = int(self.max_val - self.center // self.increment)
        left_positions = np.array([-i for i in range(num_left_coordinate)])
        right_positions = np.array([i for i in range(1, num_right_coordinate + 1)])
        self.positions = np.concatenate((left_positions, right_positions))

    def isContained(self, float_val: float) -> bool:
        """
        Check if a float value is within the range of the axis.
        
        Args:
            float_val (float): The value to check.
        
        Returns:
            bool: True if the value is within the range, False otherwise.
        """
        return self.min_val <= float_val <= self.max_val
    
    def iterate(self):
        """
        Iterate over the coordinates on the axis.
        
        Returns:
            float: The next value in the axis.
        """
        current = self.min_val  
        while current <= self.max_val:
            yield current
            current += self.increment

    def getCoordinatesAsArray(self) -> np.ndarray:
        """
        Get the coordinates of the axis based on the increment.
        
        Returns:
            np.ndarray: 1d Array of coordinates in the axis.
        """
        return np.array([self.min_val + i * self.increment for i in range(self.num_point)])
    
    def getPositionsAsArray(self) -> np.ndarray:
        """
        Get the position of the coordinates on the axis relative to its center.
        The positions are the integer index positions relative to the center of the axis.
        
        Returns:
            np.ndarray: 1d Array of integers
        """
        return np.array([self.min_val + i * self.increment for i in range(self.num_point)])
    
    def intersect(self, other: 'Axis') -> Optional['Axis']:
        """
        Find the intersection of this axis with another axis.
        
        Args:
            other (Axis): The other axis to intersect with.
        
        Returns:
            Optional[Axis]: A new Axis representing the intersection, or None if there is no intersection.
        """
        if not np.isclose(self.increment, other.increment):
            raise ValueError("Axes must have the same increment to find intersection.")
        if self.max_val < other.min_val or self.min_val > other.max_val:
            return None
        # Calculate the intersection range.
        intersect_min = max(self.min_val, other.min_val)
        intersect_max = min(self.max_val, other.max_val)
        return Axis(name=f"({self.name}-{other.name})",
                min_val=intersect_min, max_val=intersect_max, increment=self.increment)

    def __repr__(self) -> str:
        return f"{self.name}: [{self.min_val}, {self.max_val}, {self.increment}]"


class Grid(object):

    def __init__(self, centroid: Point, axes: List[Axis], name: Optional[str]=None) -> None:
        """
        
        Args:
            centroid (Point): The centroid of the grid.
            axes (List[Axis]): List of axes defining the grid.
        """
        self.centroid = centroid
        self.axes = axes
        if name is None:
            name = f"G{np.random.randint(10000, 99999)}"
        self.name = name

    def isContained(self, point: Point) -> bool:
        """
        Check if a point is contained within the grid.

        Args:
            point (Point): The point to check.

        Returns:
            bool: True if the point is within the grid, False otherwise.
        """
        return all(axis.isContained(coord) for axis, coord in zip(self.axes, point.coordinates))
    
    def __repr__(self) -> str:
        axes_str = "\n  ".join(str(axis) for axis in self.axes)
        return f"{self.name} {axes_str})"
    
    def getPointsAsArray(self) -> np.ndarray:
        """
        Get all points in the grid.
        
        Returns:
            np.ndarray: 2d Array of points in the grid.
        """
        point_arr = np.array(list(itertools.product(*[axis.getCoordinatesAsArray() for axis in self.axes])))
        return point_arr

    def intersect(self, other_grid: 'Grid') -> Union['Grid', None]:
        """
        Get all intersections of this grid with other grids.
        
        Returns:
            List[Tuple[Point, Grid]]: List of tuples containing the intersection point and the intersecting grid.
        """
        intersections = []
        for self_axis, other_axis in zip(self.axes, other_grid.axes):
            intersection_axis = self_axis.intersect(other_axis)
            if intersection_axis is None:
                return None
            intersections.append(intersection_axis)
        return Grid(self.centroid, intersections, name=f"{self.name}-{other_grid.name}")