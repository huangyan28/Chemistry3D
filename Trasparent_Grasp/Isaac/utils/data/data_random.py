import omni.replicator.core as rep

with rep.new_layer():

    seed=10
    seed2=11

    def get_shapes():
        textures = [f'/home/nvidia-3080/_out_sdrec/background/{i}.png' for i in range(1, 160)]
        shapes = rep.get.prims(path_pattern='/World/Plane')
        with shapes:
            rep.randomizer.texture(textures=textures)
        return shapes.node

    rep.randomizer.register(get_shapes)

    def get_rotates():
        rotates =  rep.get.prims(path_pattern='/World/_')
        with rotates:
            rep.modify.pose(
                rotation=rep.distribution.uniform((0, 0, -90), (0, 0, 180),seed=seed),
            )
        return rotates.node

    rep.randomizer.register(get_rotates)
    
    def rect_lights():
        lights = rep.create.light(
            light_type="Rect",
            position = (-6,0,10),
            rotation= (0,-54,90),
            intensity=rep.distribution.uniform(3000, 12000,seed=seed),
            temperature=rep.distribution.uniform(6000, 7000,seed=seed),
            scale=rep.distribution.uniform(0.05, 0.1)
            )
        return lights.node

    rep.randomizer.register(rect_lights)

    def rect_lights1():
        lights = rep.create.light(
            light_type="Rect",
            position = (8,0,10),
            rotation= (0,50,90),
            intensity=rep.distribution.uniform(3000, 12000,seed=seed2),
            temperature=rep.distribution.uniform(6000, 7000,seed=seed2),
            scale=rep.distribution.uniform(5, 10),
            )
        return lights.node

    rep.randomizer.register(rect_lights1)

    def dome_lights():
        lights = rep.create.light(
            light_type="Dome",
            rotation= (0,0,0),
            intensity=rep.distribution.uniform(200, 1000),
            temperature=rep.distribution.uniform(5000, 7500,seed=seed),
            scale=rep.distribution.uniform(5, 10),
            )
        return lights.node

    rep.randomizer.register(dome_lights)

    # Setup randomization
    with rep.trigger.on_frame(num_frames=400,interal=10）
