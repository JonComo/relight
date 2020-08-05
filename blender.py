import bpy
import bmesh
import numpy as np

# Starting Point
bpy.data.objects["Camera"].location = (10, 0, 0)
bpy.data.objects["Camera"].rotation_euler = (np.pi/2, 0, np.pi/2)
bpy.data.objects["Light"].location = (5, 0, 0)

step = 10

w = 10
h = 10

y = bpy.data.objects["Light"].location.y
z = bpy.data.objects["Light"].location.z

scene = bpy.context.scene
for i in np.arange(-w + y, w + y + 2*w/step, 2*w/step):
    for j in np.arange(-h + z, h + z + 2*h/step, 2*h/step):
        bpy.data.objects["Light"].location.y = i
        bpy.data.objects["Light"].location.z = j
        scene.render.image_settings.file_format = 'PNG'
        scene.render.filepath = "/home/billy/Documents/Berkeley/CS184/Project/dataset/circle/%f_%f.png" % (i,j)
        bpy.ops.render.render(write_still = 1)

