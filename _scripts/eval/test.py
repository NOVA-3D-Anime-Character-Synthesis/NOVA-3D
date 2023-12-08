

ckpt = ueg3d.load_eg3dc_model(inferquery, generator_module, force_sigmoid=True)
G = ckpt.G.eval().to(device)