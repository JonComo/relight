import bpy
import bmesh
import numpy as np

def rodrigues_normal(v, theta):
    normal = np.array([0, 0, 1])
    return v * np.cos(theta) + np.cross(normal, v) * np.sin(theta) + np.dot(normal, v) * normal * (1 - np.cos(theta))

# Starting Point
bpy.data.objects["Light"].location = (5, 0, 6)

step = 100
num = 0

v_original = bpy.data.objects["Light"].location.copy()
scene = bpy.context.scene
for i in np.arange(0, -np.pi/2 -np.pi/(2*step), -np.pi/(2*step)):
    v = np.array([v_original.x, v_original.y, v_original.z])
    trans = rodrigues_normal(v, i)
    x, y, z = trans[0], trans[1], trans[2]
    bpy.data.objects["Light"].location.x = x
    bpy.data.objects["Light"].location.y = y
    bpy.data.objects["Light"].location.z = z
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = "/home/billy/Documents/Berkeley/CS184/Project/dataset/ellipsoid/%f_%f_%f.png" % (x,y,z)
    bpy.ops.render.render(write_still = 1)
    num += 1
