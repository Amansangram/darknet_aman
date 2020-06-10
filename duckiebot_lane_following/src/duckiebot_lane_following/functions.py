import numpy as np
import cv2

# The homography matrix is used to project points in the image to points on the ground plane
# This is the default homography matrix from the extrinsic calibration folder
homography = np.array([[-4.89775e-05, -0.0002150858, -0.1818273],
                       [0.00099274, 1.202336e-06, -0.3280241],
                       [-0.0004281805, -0.007185673, 1]])

# The inverse of the homography matrix can be used to proints on the ground plane back into the image
homography_inverse = np.linalg.inv(homography)

# Images from the Duckiebot are 640 x 480
image_width = 640
image_height = 480

u_min = 0
u_max = image_width
v_min = image_height / 2
v_max = image_height


def line_points2slope_intercept(line_points):
    """Converts a line represented by two points to a line represented by slope and intercept

    :param line_points: An array of four values: [u1, v1, u2, v2]
    :return: An array of two values: [slope, intercept]
    """

    u1, v1, u2, v2 = line_points

    fit = np.polyfit((u1, u2), (v1, v2), 1)

    slope = fit[0]
    intercept = fit[1]

    return np.array([slope, intercept])


def line_slope_intercept2points(line_slope_intercept):
    """Converts a line represented by slope and intercept to a line represented by two points within the bottom half of
    the image

    :param line_slope_intercept: An array of two values: [slope, intercept]
    :return: An array of four values: [u1, v1, u2, v2]
    """

    slope, intercept = line_slope_intercept

    global u_min, u_max, v_min, v_max

    # Find where line intercepts bottom of image
    u1 = (v_max - intercept) / slope
    v1 = v_max

    if u1 < u_min:  # Line extends past left of image
        # Find where line intercepts left of image
        u1 = u_min
        v1 = slope * u_min + intercept
    elif u1 > u_max:  # Line extends past right of image
        # Find where line intercepts with right of image
        u1 = u_max
        v1 = slope * u_max + intercept

    # Find where line intercepts with middle of image
    u2 = (v_min - intercept) / slope
    v2 = v_min

    if u2 < u_min:  # Line extends past left of image
        # Find where line intercepts left of image
        u2 = u_min
        v2 = slope * u_min + intercept
    elif u2 > u_max:  # Line extends past right of image
        # Find where line intercepts with right of image
        u2 = u_max
        v2 = slope * u_max + intercept

    return np.array([u1, v1, u2, v2])


def point_image2ground(image_point):
    """Converts a point in the image to a point on the ground plane

    :param image_point: An array of two values: [u, v]
    :return: An array of two values: [x, y]
    """

    image_point = np.append(image_point, 1.)

    global homography

    ground_point = np.dot(homography, image_point)

    ground_point /= ground_point[2]

    return np.array([ground_point[0], ground_point[1]])


def point_ground2image(ground_point):
    """Converts a point on the ground plane to a point in the image

    :param ground_point: An array of two values: [x, y]
    :return: An array of two values: [u, v]
    """

    ground_point = np.append(ground_point, 1.)

    global homography_inverse

    image_point = np.dot(homography_inverse, ground_point)

    image_point /= image_point[2]

    return np.array([image_point[0], image_point[1]])


def line_image2ground(image_line):
    """Converts a line in the image to a line on the ground plane

    :param image_line: An array of four values: [u1, v1, u2, v2]
    :return: An array of four values: [x1, y1, x2, y2]
    """

    image_point1 = np.array([image_line[0], image_line[1]])
    image_point2 = np.array([image_line[2], image_line[3]])

    ground_point1 = point_image2ground(image_point1)
    ground_point2 = point_image2ground(image_point2)

    return np.array([ground_point1[0], ground_point1[1], ground_point2[0], ground_point2[1]])


def line_ground2image(ground_line):
    """Converts a line on the ground plane to a line in the image

    :param ground_line: An array of four values: [x1, y1, x2, y2]
    :return: An array of four values: [u1, v1, u2, v2]
    """

    ground_point1 = np.array([ground_line[0], ground_line[1]])
    ground_point2 = np.array([ground_line[2], ground_line[3]])

    image_point1 = point_ground2image(ground_point1)
    image_point2 = point_ground2image(ground_point2)

    line_image_points = np.array([image_point1[0], image_point1[1], image_point2[0], image_point2[1]])

    # Convert to slope, intercept and back to force points to be within image
    line_slope_intercept = line_points2slope_intercept(line_image_points)
    line_image_points_from_slope_intercept = line_slope_intercept2points(line_slope_intercept)

    global u_min, u_max, v_min, v_max

    if line_image_points[0] < u_min or line_image_points[0] > u_max or \
            line_image_points[1] < v_min or line_image_points[1] > v_max:
        # Replace first point from first point converted from slope and intercept
        line_image_points[0] = line_image_points_from_slope_intercept[0]
        line_image_points[1] = line_image_points_from_slope_intercept[1]

    if line_image_points[2] < u_min or line_image_points[2] > u_max or \
            line_image_points[3] < v_min or line_image_points[3] > v_max:
        # Replace first point from first point converted from slope and intercept
        line_image_points[2] = line_image_points_from_slope_intercept[2]
        line_image_points[3] = line_image_points_from_slope_intercept[3]

    return line_image_points


def average_slope_intercept(line_segments):
    if line_segments is None:
        return [None, None]

    left_fit = []
    right_fit = []

    # boundary = 1 / 3
    boundary = 0.5
    left_region_boundary = image_width * (1 - boundary)  # left lane line segment should be on left 2/3 of the screen
    right_region_boundary = image_width * boundary  # right lane line segment should be on left 2/3 of the screen

    for line_segment in line_segments:

        u1 = line_segment[0]
        u2 = line_segment[2]

        if u1 == u2:
            continue

        try:
            [slope, intercept] = line_points2slope_intercept(line_segment)
        except:
            continue

        if slope < -0.5 and u1 < left_region_boundary and u2 < left_region_boundary:
            # Left lane
            left_fit.append((slope, intercept))
        elif slope > 0.5 and u1 > right_region_boundary and u2 > right_region_boundary:
            # Right lane
            right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)

    left_lane = None
    right_lane = None

    if len(left_fit) > 0:
        left_lane = line_slope_intercept2points(left_fit_average)

    if len(right_fit) > 0:
        right_lane = line_slope_intercept2points(right_fit_average)

    return [left_lane, right_lane]


def plot_image_lines(image_lines):
    """Plot image coordinate lines onto image

    :param image_lines: An array of arrays of four values: [[u1, v1, u2, v2], [u1, v1, u2, v2]]
    :return: image
    """

    global image_height, image_width

    # Initialise blank image
    image = np.zeros([image_height, image_width], np.uint8)

    # Draw lines on image
    for image_line in image_lines:
        u1, v1, u2, v2 = image_line
        cv2.line(image, (int(u1), int(v1)), (int(u2), int(v2)), 255, 1, cv2.LINE_AA)

    return image


def plot_ground_lines(ground_lines):
    """Plot ground coordinate lines onto image

    :param ground_lines: An array of arrays of four values: [[x1, y1, x2, y2], [x1, y1, x2, y2]]
    :return: image
    """
    # Convert to image lines
    image_lines = []

    for ground_line in ground_lines:
        image_lines.append(line_ground2image(ground_line))

    return plot_image_lines(image_lines)
