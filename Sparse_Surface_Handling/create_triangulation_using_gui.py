"""
Created on Sep 10 08:38

@author: ISAC - pettirsch
"""

import argparse
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from Utils.PerspectiveTransform.perspectiveTransform import PerspectiveTransform

import matplotlib
matplotlib.use('TkAgg')

parser = argparse.ArgumentParser()
parser.add_argument('--gtFaces', type=str,
                    default='roadsurfaceTriangulationFaces.csv',
                    help='json config path')
parser.add_argument('--gtPoints', type=str,
                    default='roadsurfaceTriangulationPoints.csv',
                    help='json config path')
parser.add_argument('--calibrationPath', type=str,
                    default='/calibrationMatrix.xml',
                    help='json config path')
parser.add_argument('--outputpath',
                    default='',
                    help='save to project/name')
parser.add_argument("--filePath", help="path to video or image file",
                    default="Test.png")
opt = parser.parse_args()


def open_image(file_path):
    # Check if the input path is a video or an image
    if file_path.endswith(('.mp4', '.mov', '.avi')):
        # Open Video with OpenCV
        cap = cv2.VideoCapture(file_path)
        # get the first frame
        ret, image = cap.read()
    elif file_path.endswith(('.jpg', '.png', '.jpeg')):
        # Open Image with OpenCV
        image = cv2.imread(file_path)
    else:
        # Raise error if the input path is neither a video nor an image
        raise ValueError(
            'The input path is neither a video nor an image. Please provide a mp4, mov, avi, jpg, png or jpeg file.')

    return image


def click_event(event, x, y, flags, param):
    global curr_triangle

    if event == cv2.EVENT_LBUTTONDOWN:
        if not define_triangle:
            point_3d = perspectiveTransform.pixelToStreePlane(np.array((x, y)))
            if np.min(point_3d) > 0:
                clicked_points_3d.append(point_3d)
                clicked_points_2d.append((x, y))
                cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            else:
                # If the point is not on the street plane, print an error message
                print("The clicked point is not on the street plane. Please click on a point on the street plane.")
            cv2.imshow('image', image)
        else:
            # find the closest point to the clicked point in clicked_points_2d_np
            clicked_point = np.array([x, y])
            distances = np.linalg.norm(clicked_points_2d_np - clicked_point, axis=1)
            closest_point_idx = np.argmin(distances)
            curr_triangle.append(closest_point_idx)
            if len(curr_triangle) == 3:
                # Create a blank mask
                mask = np.zeros_like(image, dtype=np.uint8)
                triangle = np.array([clicked_points_2d[curr_triangle[0]], clicked_points_2d[curr_triangle[1]],
                                        clicked_points_2d[curr_triangle[2]]])
                cv2.fillPoly(mask, np.array([triangle]), (255, 255, 255), lineType=cv2.LINE_AA)
                # Apply the mask to the image (overlay the polygon)
                cv2.addWeighted(mask, 0.5, image, 1, 0, image)

                # Get random color
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.polylines(image, [triangle], True, color, 2)

                faces.append(curr_triangle)
                curr_triangle = []

def create_single_plane_3_points(pointA, pointB, pointC):
    # Create a plane using 3 points
    # Calculate the normal vector of the plane
    normal_vector = np.cross(pointB - pointA, pointC - pointA)
    # Calculate the distance from the origin to the plane
    d = -np.dot(normal_vector, pointA)
    # Normalize the normal vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    # Return the normal vector and the distance
    return normal_vector, d


def save_point_cloud_to_csv(points, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        for point in points:
            writer.writerow(point)


class PlaneModelPCA:
    def fit(self, X):
        # Step 1: Subtract the mean from the data to center it
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Step 2: Compute the covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)

        # Step 3: Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Step 4: The normal to the plane is the eigenvector corresponding to the smallest eigenvalue
        self.normal_vector = eigenvectors[:, 0]

        # The plane coefficients (a, b, c, d) in ax + by + cz + d = 0
        self.coef_ = np.append(self.normal_vector, -np.dot(self.normal_vector, self.mean_))

        return self

    def predict(self, X):
        # Use the plane equation ax + by + cz + d = 0 to calculate z for given x, y
        # We rearrange this to z = (-d - ax - by) / c
        z = -(self.coef_[0] * X[:, 0] + self.coef_[1] * X[:, 1] + self.coef_[3]) / self.coef_[2]
        return z


def get_surface_interactive(path_gt_faces=None, path_gt_points=None, calibrationPath=None, output_path=None):
    # Use global variables
    global image
    global clicked_points_2d
    global clicked_points_3d
    global perspectiveTransform
    global faces
    global curr_triangle
    global define_triangle
    global clicked_points_2d_np
    define_triangle= False
    clicked_points_2d = []
    clicked_points_3d = []
    faces = []
    curr_triangle = []

    # Load GT data
    gt_faces = np.loadtxt(path_gt_faces, delimiter=",")
    gt_points = np.loadtxt(path_gt_points, delimiter=",")

    # Load perspective transformation
    perspectiveTransform = PerspectiveTransform()
    perspectiveTransform.loadCalibrationFromXML(calibrationPath)
    perspectiveTransform.loadTriangulationPointsFromCSV(path_gt_points)
    perspectiveTransform.loadTriangulationFacesFromCSV(path_gt_faces)
    perspectiveTransform.initCalibration()

    # Load image
    image = open_image(opt.filePath)

    # Create a window and set the callback function
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback('image', click_event)

    while True:
        cv2.imshow('image', image)

        if cv2.waitKey(1) & 0xFF == ord('t'):
            define_triangle = not define_triangle
            if define_triangle:
                clicked_points_2d_np = np.array(clicked_points_2d)
                clicked_points_2d_np = np.reshape(clicked_points_2d_np, (clicked_points_2d_np.shape[0], 2))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Create surface
    plot = True
    if len(clicked_points_3d) == 3:
        # Create single plane
        normal_vector, d = create_single_plane_3_points(clicked_points_3d[0], clicked_points_3d[1],
                                                        clicked_points_3d[2])
        plane_coefficients = np.array([normal_vector[0], normal_vector[1], normal_vector[2], d])
        planeCoefficientsPath = os.path.join(output_path, "plane_coefficients.csv")
        np.savetxt(planeCoefficientsPath, plane_coefficients, delimiter=",")
        plane_point = clicked_points_3d[0]
        plane_point = np.reshape(plane_point, (1, 3))
        planePointPath = os.path.join(output_path, "plane_points.csv")
        save_point_cloud_to_csv(plane_point, planePointPath)

        faces = None
    elif len(clicked_points_3d) > 3:
        # Save fitted plane
        points_3d = np.array(clicked_points_3d)
        points_3d = np.reshape(points_3d, (points_3d.shape[0], 3))
        plane_model = PlaneModelPCA()
        plane_model.fit(points_3d)
        plane_coefficients = plane_model.coef_
        planeCoefficientsPath = os.path.join(output_path, "plane_coefficients.csv")
        np.savetxt(planeCoefficientsPath, plane_coefficients, delimiter=",")
        plane_point = clicked_points_3d[0]
        plane_point = np.reshape(plane_point, (1, 3))
        planePointPath = os.path.join(output_path, "plane_points.csv")
        save_point_cloud_to_csv(plane_point, planePointPath)

        # Save triangulated plane
        # tri = Delaunay(points_3d[:, :2])
        vertices = points_3d
        # faces = tri.simplices
        # Save vertices to a CSV file
        vertices_df = pd.DataFrame(vertices, columns=['X', 'Y', 'Z'])
        vertices_df.to_csv(f"{output_path}/roadsurfaceTriangulationPoints.csv", index=False, header=False)
        # Save faces to a CSV file
        faces_df = pd.DataFrame(faces, columns=['Vertex1', 'Vertex2', 'Vertex3'])
        faces_df.to_csv(f"{output_path}/roadsurfaceTriangulationFaces.csv", index=False, header=False)
    else:
        print("Please click on at least 3 points to create a surface.")
        plot = False

    if plot:

        # Save the image with the clicked points
        cv2.imwrite(os.path.join(output_path, 'clicked_points.png'), image)

        # Create matplotlib figure and axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('East')
        ax.set_ylabel('North')
        ax.set_zlabel('Z')

        # Plot clicked points
        if clicked_points_3d:
            clicked_points_3d = np.array(clicked_points_3d)
            clicked_points_3d = np.reshape(clicked_points_3d, (clicked_points_3d.shape[0], 3))
            ax.scatter(clicked_points_3d[:, 0], clicked_points_3d[:, 1], clicked_points_3d[:, 2], c='r', marker='o')

        # Generate a grid of points to plot the plane
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x = np.linspace(xlim[0] - 10, xlim[1] + 10, 10)
        y = np.linspace(ylim[0] - 10, ylim[1] + 10, 10)
        x, y = np.meshgrid(x, y)

        # Plt faces
        if faces is not None:
            for face in faces:
                for i in range(3):
                    ax.plot([vertices[face[i], 0], vertices[face[(i + 1) % 3], 0]],
                            [vertices[face[i], 1], vertices[face[(i + 1) % 3], 1]],
                            [vertices[face[i], 2], vertices[face[(i + 1) % 3], 2]], 'b-', lw=1)

        # Plot single plane
        z = -(plane_coefficients[0] * x + plane_coefficients[1] * y + plane_coefficients[3]) / plane_coefficients[2]
        ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100, color='green')

        # Plot GT points
        for face in gt_faces:
            for i in range(3):
                if gt_points[int(face[i]), 0] > xlim[0] and gt_points[int(face[i]), 0] < xlim[1] and gt_points[
                    int(face[i]), 1] > ylim[0] and gt_points[int(face[i]), 1] < ylim[1]:
                    plot_gt = True
                else:
                    plot_gt = False

            if plot_gt:
                for i in range(3):
                    ax.plot([gt_points[int(face[i]), 0], gt_points[int(face[(i + 1) % 3]), 0]],
                            [gt_points[int(face[i]), 1], gt_points[int(face[(i + 1) % 3]), 1]],
                            [gt_points[int(face[i]), 2], gt_points[int(face[(i + 1) % 3]), 2]], 'k-', lw=1)

        # Open interactive plot
        plt.show()

        ax.figure.canvas.draw()
        img = np.frombuffer(ax.figure.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(ax.figure.canvas.get_width_height()[::-1] + (3,))

        non_white_pixels = np.where(np.any(img != 255, axis=-1))
        ymin, ymax = np.min(non_white_pixels[0]), np.max(non_white_pixels[0])
        xmin, xmax = np.min(non_white_pixels[1]), np.max(non_white_pixels[1])

        # Crop the image to the bounding box
        img_cropped = img[ymin:ymax + 1, xmin:xmax + 1]

        # Ensure that img_cropped has the same aspect ratio as 640 x 480 and resize it
        h, w, _ = img_cropped.shape

        plt.imsave(os.path.join(output_path, 'surface_fit.png'), img_cropped)
        plt.close()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    get_surface_interactive(path_gt_faces=opt.gtFaces, path_gt_points=opt.gtPoints, calibrationPath=opt.calibrationPath,
                            output_path=opt.outputpath)
