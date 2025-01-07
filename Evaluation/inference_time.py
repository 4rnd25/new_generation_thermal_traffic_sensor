"""
Created on 11 04 2024 16:00

@author: ISAC - pettirsch
"""

from PIL import Image
import onnxruntime as ort
import numpy as np
import time

# Parameter
providers = ['CUDAExecutionProvider','CPUExecutionProvider']
model_path = ""
input_image = "/"
buttom_map = ""
cam_pos_path = ""
input_size = 640

# Define class mean values
class_means = {
    "Motorcycle": {'Length': 1.8181932773200586, 'Width': 0.8580672268856184, 'Height': 1.5351680672268904},
    "Car": {'Length': 3.644727143871292, 'Width': 1.8174202693149704, 'Height': 1.38774881036752},
    "Truck": {'Length': 7.469069767434114, 'Width': 2.5868992248377958, 'Height': 2.6714211886304926},
    "Bus": {'Length': 13.453127753321924, 'Width': 2.940550660784477, 'Height': 2.7024008810572684},
    "Person": {'Length': 0.7048374207757256, 'Width': 0.6764811242639799, 'Height': 1.5817759713419663},
    "Bicycle": {'Length': 1.6588440401753517, 'Width': 0.7437218116358572, 'Height': 1.5825260564714794},
    "E-Scooter": {'Length': 1.271192411936881, 'Width': 0.5704336043463357, 'Height': 1.650677506775067}
}

# Create a NumPy array for the mean values
cls_mean_array = np.array([
    [values['Length'], values['Width'], values['Height']] for values in class_means.values()
])


def speedtest(model_path, providers, input_size):
    # Create onnxruntime session
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(model_path, providers=providers)

    # Create input tensor from image
    image = Image.open(input_image)
    # Convert the image to a NumPy array
    image = np.array(image)
    image = np.expand_dims(image, axis=0)

    # Transpose the image to the correct format
    image = np.transpose(image, (0, 2, 1, 3))

    # Pad image of size 640x480 to 640x640
    pad = (640 - 480) // 2
    image = np.pad(image, ((0, 0), (0, 0), (pad, pad), (0, 0)), mode='constant')

    # Transpose the image to the correct format
    image = np.transpose(image, (0, 3, 2, 1))

    # transfer image to float32
    image = image.astype(np.float16)

    # Load bottom map and camera position
    bottom_map = np.load(buttom_map)
    cam_pos = np.load(cam_pos_path)

    times_inf = []
    times_post = []

    for i in range(1000):
        if i == 10:
            start = time.time()

        # Perform your model inference here
        outputs = sess.run(None, {"images": image})

        xyxy = outputs[0][:,0:4]
        obj_conf = outputs[0][:,4]
        cls_conf = outputs[0][:,5:12]
        kpt_img = outputs[0][:,12:14]
        dim = outputs[0][:,14:17]
        obs_angle = outputs[0][:,17:19]

        # Get max score and class
        max_score = np.max(obj_conf.reshape(100, 1) * cls_conf, axis=1)
        max_class = np.argmax(obj_conf.reshape(100, 1) * cls_conf, axis=1)

        # Create new numpy array [xyxy, conf, cls, kpt, d_length, d_width, d_height, s_angle, c_angle]
        outputs = np.column_stack([xyxy, max_score, max_class, kpt_img, dim, obs_angle])

        outputs = [outputs]

        # Create 3D Boxes
        boxes = calc_3d_output_numpy(outputs, bottom_map, cam_pos, cls_mean_array)

    end = time.time()

    elapsed_time = end - start
    mean_time_inf = elapsed_time / 990


    print(f"Mean inference time: {mean_time_inf} s")


    return mean_time_inf

def calc_3d_output_numpy(pred, bottom_map, cam_pos, cls_mean_lookup, img_shape=(640, 480)):
    """
    Optimized version of calc_3d_output_numpy.
    """
    # Stack predictions from all images and add image index
    pred_all = np.vstack([np.hstack([p, np.full((p.shape[0], 1), i)]) for i, p in enumerate(pred)])

    if pred_all.shape[0] == 0:
        return []

    # Scale 2D boxes
    scale_coords_numpy((bottom_map.shape[1], bottom_map.shape[0]), pred_all[:, :4], img_shape)

    # Extract bottom center image, dimensions, and angles
    pred_bc_img = pred_all[:, 6:8]
    pred_dim = pred_all[:, 8:11]
    pred_obs_angle = pred_all[:, 11:13]

    # Handle padding
    gain = min(img_shape[0] / bottom_map.shape[1], img_shape[1] / bottom_map.shape[0])
    pad = ((img_shape[1] - bottom_map.shape[0] * gain) / 2, (img_shape[0] - bottom_map.shape[1] * gain) / 2)
    pred_bc_img -= np.array(pad)

    # Filter bottom center image coordinates
    pred_bc_img = np.round(pred_bc_img).astype(int)
    valid_mask = (
        (pred_bc_img[:, 0] >= 0) & (pred_bc_img[:, 0] < bottom_map.shape[0]) &
        (pred_bc_img[:, 1] >= 0) & (pred_bc_img[:, 1] < bottom_map.shape[1])
    )
    pred_bc_img_valid = pred_bc_img[valid_mask]
    pred_dim_valid = pred_dim[valid_mask]
    pred_obs_angle_valid = pred_obs_angle[valid_mask]

    # Get BC world
    pred_bc_world = bottom_map[pred_bc_img_valid[:, 0], pred_bc_img_valid[:, 1], :] + cam_pos.squeeze()
    in_bounds_mask = (pred_bc_world >= 0).all(axis=1)
    pred_bc_world_valid = pred_bc_world[in_bounds_mask]
    pred_dim_valid = pred_dim_valid[in_bounds_mask]
    pred_obs_angle_valid = pred_obs_angle_valid[in_bounds_mask]

    # Compute yaw angles
    obs_angle = np.arctan2(pred_obs_angle_valid[:, 0], pred_obs_angle_valid[:, 1])
    atan_xy = np.arctan2(pred_bc_world_valid[:, 0] - cam_pos[0],
                         pred_bc_world_valid[:, 1] - cam_pos[1])
    yaw_angle = atan_xy + obs_angle

    # Compute 3D dimensions
    delta_dims = np.exp(pred_dim_valid)
    mean_dims = cls_mean_lookup[pred_all[:, 5].astype(int)[valid_mask][in_bounds_mask]]
    dims = delta_dims * mean_dims

    # Final valid predictions
    xmin, ymin, xmax, ymax = np.split(pred_all[valid_mask][in_bounds_mask][:, :4], 4, axis=1)
    conf = pred_all[valid_mask][in_bounds_mask][:, 4]
    cls = pred_all[valid_mask][in_bounds_mask][:, 5].astype(int)

    output = np.hstack((
        xmin, ymin, xmax, ymax, conf[:, None], cls[:, None],
        pred_bc_img_valid[in_bounds_mask],
        pred_bc_world_valid,
        dims,
        yaw_angle[:, None]
    ))

    # Group by image index
    image_indices = pred_all[valid_mask][in_bounds_mask][:, 13].astype(int)
    output_list = [output[image_indices == i] for i in range(len(pred))]

    return output_list


def scale_coords_numpy(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    Optimized scaling of coordinates from img1_shape to img0_shape.
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (
            (img1_shape[1] - img0_shape[1] * gain) / 2,
            (img1_shape[0] - img0_shape[0] * gain) / 2
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords_numpy(coords, img0_shape)


def clip_coords_numpy(boxes, img_shape):
    """
    Optimized bounding box clipping to image dimensions.
    """
    np.clip(boxes[:, [0, 2]], 0, img_shape[1], out=boxes[:, [0, 2]])  # x1, x2
    np.clip(boxes[:, [1, 3]], 0, img_shape[0], out=boxes[:, [1, 3]])  # y1, y2


if __name__ == "__main__":
    mean_time = speedtest(model_path, providers, input_size)