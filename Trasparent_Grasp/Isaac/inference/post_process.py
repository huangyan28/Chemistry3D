from skimage.filters import gaussian


def post_process_output(q_img, radius_img):
    q_img = q_img.cpu().numpy().squeeze()
    radius_img = radius_img.cpu().numpy().squeeze() * 112
    q_img = gaussian(q_img, 2.0, preserve_range=True)
    radius_img = gaussian(radius_img, 1.0, preserve_range=True)
    return q_img, radius_img
