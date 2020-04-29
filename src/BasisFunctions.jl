using Makie

cmp = :pu_or
# sheen = 32.0f0
sheen = 16.0f0
camera_translation = (0.0, 0.0, 0.0)
camera_rotation = (0.0, 0.2, 0.0)

# xlims = [-0.1,  1.1, -0.1,  1.1, -0.1,  1.1, -0.1,  1.1]
# ylims = [-0.1,  1.1,  1.1, -0.1,  1.1,  1.1, -0.1, -0.1]
# zlims = [ 0.6,  0.6,  0.6,  0.6, -0.6, -0.6, -0.6, -0.6]

lim = FRect3D((-0.1, -0.1, -0.6), (1.2, 1.2, 1.2))


ϕ01(x,y) = 0.5 * Float32(0 < x < 1) * Float32(0 <  y <  1)

ϕ11(x,y) = 0.5  * Float32(0 < x < 1/2) * Float32(0 < y < 1/2)
ϕ12(x,y) = 0.5  * Float32(1/2 < x < 1) * Float32(0 < y < 1/2)
ϕ13(x,y) = 0.5  * Float32(0 < x < 1/2) * Float32(1/2 < y < 1)
ϕ14(x,y) = 0.5  * Float32(1/2 < x < 1) * Float32(1/2 < y < 1)

ϕ21(x,y) = 0.5  * Float32(2/4 < x < 3/4) * Float32(2/4 < y < 3/4)
ϕ22(x,y) = 0.5  * Float32(3/4 < x < 4/4) * Float32(2/4 < y < 3/4)
ϕ23(x,y) = 0.5  * Float32(2/4 < x < 3/4) * Float32(3/4 < y < 4/4)
ϕ24(x,y) = 0.5  * Float32(3/4 < x < 4/4) * Float32(3/4 < y < 4/4)


vx = -0.01: 0.0103 : 1.01
vy = -0.01: 0.0103 : 1.01


plot1 = Scene();
z = Float32[ϕ01(x,y) for x in vx, y in vy];
surface!(plot1, vx, vy, z, colormap=cmp, shininess=sheen, show_axis=false, shading=false, colorrange=(-0.5, 0.5), limits=lim);
translate_cam!(plot1, camera_translation);
rotate_cam!(plot1, camera_rotation);

plot2 = Scene();
z = Float32[ϕ11(x,y) + ϕ12(x,y) - ϕ13(x,y) - ϕ14(x,y) for x in vx, y in vy];
surface!(plot2, vx, vy, z, colormap=cmp, show_axis=false, shading=false, colorrange=(-0.5, 0.5), limits=lim);
translate_cam!(plot2, camera_translation);
rotate_cam!(plot2, camera_rotation);

plot3 = Scene();
z = Float32[ϕ11(x,y) - ϕ12(x,y) + ϕ13(x,y) - ϕ14(x,y) for x in vx, y in vy];
surface!(plot3, vx, vy, z, colormap=cmp, show_axis=false, colorrange=(-0.5, 0.5), limits=lim);
translate_cam!(plot3, camera_translation);
rotate_cam!(plot3, camera_rotation);

plot4 = Scene();
z = Float32[ϕ11(x,y) - ϕ12(x,y) - ϕ13(x,y) + ϕ14(x,y) for x in vx, y in vy];
surface!(plot4, vx, vy, z, colormap=cmp, show_axis=false, colorrange=(-0.5,0.5), limits=lim);
translate_cam!(plot4, camera_translation);
rotate_cam!(plot4, camera_rotation);

plot5 = Scene();
z = Float32[ϕ21(x,y) + ϕ22(x,y) - ϕ23(x,y) - ϕ24(x,y) for x in vx, y in vy];
surface!(plot5, vx, vy, z, colormap=cmp, show_axis=false, colorrange=(-0.5, 0.5), limits=lim);
translate_cam!(plot5, camera_translation);
rotate_cam!(plot5, camera_rotation);

plot6 = Scene();
z = Float32[ϕ21(x,y) - ϕ22(x,y) + ϕ23(x,y) - ϕ24(x,y) for x in vx, y in vy];
surface!(plot6, vx, vy, z, colormap=cmp, show_axis=false, colorrange=(-0.5, 0.5), limits=lim);
translate_cam!(plot6, camera_translation);
rotate_cam!(plot6, camera_rotation);

plot7 = Scene();
z = Float32[ϕ21(x,y) - ϕ22(x,y) - ϕ23(x,y) + ϕ24(x,y) for x in vx, y in vy];
surface!(plot7, vx, vy, z, colormap=cmp, show_axis=false, colorrange=(-0.5, 0.5), limits=lim);
translate_cam!(plot7, camera_translation);
rotate_cam!(plot7, camera_rotation);

out_scene = vbox(plot1, plot2, plot3, plot4, plot5, plot6, plot7, parent=Scene(resolution=(2400, 400), clear=false))
save("./out/plots/basis_functions.png", out_scene)