"""Finite A* on actually scanned free space; unknown cells remain blocked."""
import heapq
import math

import numpy as np
from scipy.ndimage import binary_dilation


class ScanMap:
    def __init__(self,extent=2.,resolution=.04,radius=.23):
        self.extent=extent;self.resolution=resolution;self.radius=radius
        self.size=int(2*extent/resolution)+1
        self.free=np.zeros((self.size,self.size),bool);self.occupied=self.free.copy()

    def cell(self,xy):return tuple(np.rint((np.asarray(xy)+self.extent)/self.resolution).astype(int))
    def inside(self,cell):return all(0<=x<self.size for x in cell)
    def world(self,cell):return np.asarray(cell)*self.resolution-self.extent

    def add(self,origin,directions,ranges):
        origin=np.asarray(origin,float)
        for direction,distance in zip(directions,ranges):
            if distance is None or not np.isfinite(distance) or distance<.12:continue
            direction=np.asarray(direction,float);end=origin+direction*distance
            for length in np.arange(0.,max(0.,distance-self.resolution),self.resolution/2):
                cell=self.cell(origin+direction*length)
                if self.inside(cell):self.free[cell]=True
            hit=self.cell(end)
            if self.inside(hit):self.occupied[hit]=True

    def block_box(self,lower,upper):
        lo,hi=self.cell(lower),self.cell(upper)
        for x in range(max(0,lo[0]),min(self.size,hi[0]+1)):
            for y in range(max(0,lo[1]),min(self.size,hi[1]+1)):self.occupied[x,y]=True

    def add_current_scans(self,scans):
        """Clear a stale hit only where CURRENT measured rays pass through.

        Merge all scanners before clearing: one sensor cannot erase another
        sensor's positive return. Hit cells and their grid-quantization border
        take precedence over a grazing free ray. Unknown space stays blocked.
        """
        observed=ScanMap(self.extent,self.resolution,self.radius)
        for origin,directions,ranges in scans:observed.add(origin,directions,ranges)
        hit_border=binary_dilation(observed.occupied,iterations=1)
        self.occupied[observed.free&~hit_border]=False
        self.occupied|=observed.occupied;self.free|=observed.free

    def route(self,start,goal):
        radius=int(math.ceil(self.radius/self.resolution));x,y=np.mgrid[-radius:radius+1,-radius:radius+1]
        structure=x*x+y*y<=radius*radius
        blocked=binary_dilation(self.occupied,structure=structure)
        # A centre is usable only when its entire footprint was observed free.
        unknown=~self.free;blocked|=binary_dilation(unknown,structure=structure,border_value=1)
        start,goal=self.cell(start),self.cell(goal)
        if not self.inside(start) or not self.inside(goal) or blocked[goal]:raise ValueError('Маршрут или цель не подтверждены сканированием')
        if blocked[start]:raise ValueError('Текущая опорная область не подтверждена сканированием')
        queue=[(0.,0.,start)];cost={start:0.};parents={}
        while queue:
            _,distance,current=heapq.heappop(queue)
            if current==goal:
                path=[current]
                while current!=start:current=parents[current];path.append(current)
                return np.array([self.world(c) for c in reversed(path)])
            for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
                nxt=(current[0]+dx,current[1]+dy)
                if not self.inside(nxt) or blocked[nxt]:continue
                new=distance+self.resolution
                if new<cost.get(nxt,float('inf')):
                    cost[nxt]=new;parents[nxt]=current
                    heapq.heappush(queue,(new+np.linalg.norm(np.array(nxt)-goal)*self.resolution,new,nxt))
        raise ValueError('Нет пути через наблюдаемое свободное пространство')
