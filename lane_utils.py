import cv2
import pickle
import peakutils # Find peaks
import scipy.misc
import numpy as np
# import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from moviepy.editor import VideoFileClip


def image_unwrap(image, objpoints, imgpoints, src, dst):
    # Calibrate camera and undistort image
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1::-1], None, None)
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    # Find 4 points and then wrap the image
    if ret != 0:
        # Given src and dst points, calculate the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)
        rev_M = cv2.getPerspectiveTransform(dst, src)
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(undist, M, (image.shape[1],image.shape[0]))
    else:
        print('Camera calibration failed.')

    return warped, M, rev_M

def abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_thresh(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)

    # Apply threshold
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return mag_binary

def dir_thresh(gray, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Apply threshold
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary

def color_thresh(image, r_thresh=(0, 255), s_thresh=(0, 255)):
    # Convert image to hls
    # hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    r_channel = image[:,:,0]
    # s_channel = hls[:,:,2]

    # Apply R channel threshold
    r_image = np.zeros_like(r_channel)
    r_image[(r_channel >= r_thresh[0]) & (r_channel <= r_thresh[1])] = 1
    # scipy.misc.imsave('output_images/out_rthresh.png', r_image)

    # Apply S channel threshold
    # s_image = np.zeros_like(s_channel)
    # s_image[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 255
    # scipy.misc.imsave('output_images/out_sthresh.png', s_image)

    # Apply both channel threshold
    # rs_image = np.zeros_like(r_channel)
    # rs_image[((r_channel>=r_thresh[0]) & (r_channel<=r_thresh[1])) | ((s_channel>=s_thresh[0]) & (s_channel<=s_thresh[1]))] = 255
    # scipy.misc.imsave('output_images/out_rsthresh.png', rs_image)
    return r_image

def find_lane_pixels(binary_warped, nwindows, margin, minpix):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # plt.plot(histogram)
    # plt.show()

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    peaks = peakutils.indexes(np.int32(histogram), thres=10, min_dist=150, thres_abs=True)
    print("peaks", peaks)
    print(histogram[peaks])

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows

    # Create empty lists to receive left and right lane pixel indices
    x_array = []
    y_array = []

    for peak in peaks:
        print(peak)
        lane_inds = []
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = peak - margin
            win_x_high = peak + margin

            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_x_low,win_y_low),
            (win_x_high,win_y_high),(0,255,0), 5)

            # Identify the nonzero pixels in x and y within the window #
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the list
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                peak = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices (previously was a list of lists of pixels)
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        print("x std", np.std(x))
        print("y std", np.std(y))
        print()

        # Save results to x, y arrays
        x_array.append(x)
        y_array.append(y)

    return x_array, y_array, out_img


def fit_polynomial(binary_warped, nwindows, margin, minpix, color=(0,0,255), thickness=8):
    # Find our lane pixels first
    x_array, y_array, out_img = find_lane_pixels(binary_warped, nwindows, margin, minpix)

    for x, y in zip(x_array,y_array):
        # Fit a second order polynomial to each using `np.polyfit` ###
        fit = np.polyfit(y, x, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        try:
            fitx = fit[0]*ploty**2 + fit[1]*ploty + fit[2]
        except TypeError:
            # Avoids an error if `left` and `right_fit` are still none or incorrect
            print('The function failed to fit a line!')
            fitx = 1*ploty**2 + 1*ploty

        # Colors in the lane regions
        out_img[y, x] = [0, 0, 255]
        cv2.polylines(img=out_img, pts=np.int32([np.stack((fitx, ploty), axis=-1)]), isClosed=False, color=color, thickness=thickness)

    return out_img


def _detect_lanes(image, src, dst, abs_kernel, abs_thresh, mag_kernel, mag_thresh, dir_kernel, dir_thresh, r_thresh, s_thresh, nwindows, margin, minpix, line_color, line_thicknesss):
    # Unwrap image to bird's eye view
    unwrap, perspective_M, rev_M = image_unwrap(image, objpoints, imgpoints, src, dst)

    # Gray scale image
    # gray = cv2.cvtColor(unwrap, cv2.COLOR_RGB2GRAY)

    # Apply each of the thresholding functions
    # gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=3, thresh=(20, 100))
    # grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=3, thresh=(20, 100))
    # mag_binary = mag_thresh(gray, sobel_kernel=9, mag_thresh=(30, 100))
    # dir_binary = dir_thresh(gray, sobel_kernel=15, thresh=(-np.pi/20, np.pi/20))
    color_image = color_thresh(unwrap, r_thresh=r_thresh, s_thresh=s_thresh)

    # Combination of different thresholds
    # combined = np.zeros_like(dir_binary)
    # combined[(gradx==1) & (color_image==255)] = 255
    #
    # gradxy = np.zeros_like(dir_binary)
    # gradxy[((gradx == 1)|(grady == 1))] = 1
    # magdir = np.zeros_like(dir_binary)
    # magdir[(mag_binary==1)&(dir_binary==1)] = 1
    #
    scipy.misc.imsave('output_images/unwrap.png', unwrap)
    scipy.misc.imsave('output_images/color_thres.png', color_image*255)
    # scipy.misc.imsave('output_images/out_gradxy.png', gradxy)
    # scipy.misc.imsave('output_images/out_dir.png', dir_binary)
    # scipy.misc.imsave('output_images/out_mag.png', mag_binary)
    # scipy.misc.imsave('output_images/out_gradX.png', gradx*255)
    # scipy.misc.imsave('output_images/out_gradY.png', grady*255)

    # Use sliding windows to fit the lane line
    lines_image = fit_polynomial(color_image, nwindows, margin, minpix, line_color, line_thicknesss)
    # scipy.misc.imsave('output_images/lines.png', lines_image)

    # Wrap image back
    newwarp = cv2.warpPerspective(lines_image, rev_M, (image.shape[1], image.shape[0]))
    scipy.misc.imsave('output_images/annotated.png', newwarp)

    # Merge the lane line image to real image
    result = cv2.addWeighted(image, 1, newwarp, 1, 0)
    scipy.misc.imsave('output_images/final.png', result)

    return result

def detect_lanes(image):
    # Variables
    src = np.float32([[575,464], [707,464], [258,682], [1049,682]])
    dst = np.float32([[450,0], [830,0], [450,720], [830,720]])

    abs_kernel = 3
    abs_thresh = (20,100)
    mag_kernel = 9
    mag_thresh = (30,100)
    dir_kernel = 15
    dir_thresh = (-np.pi/20, np.pi/20)
    red_thresh = (200,255)
    sat_thresh = (200,255)

    nwindows = 5 # Choose the number of sliding windows
    margin = 30 # Set the width of the windows +/- margin
    minpix = 15 # Set minimum number of pixels found to recenter window

    line_color = (255, 255, 0) # default yellow
    line_thicknesss = 8 # unit: pixel

    return _detect_lanes(image, src, dst, abs_kernel, abs_thresh, mag_kernel, mag_thresh, dir_kernel, dir_thresh, red_thresh, sat_thresh, nwindows, margin, minpix, line_color, line_thicknesss)

# clip = VideoFileClip("project_video.mp4")
# clip_out = clip.fl_image(draw_line)
# clip_out.write_videofile("output_images/out_video.mp4", audio=False)

if __name__ == '__main__':
    # Read images
    # straight = mpimg.imread('test_images/straight_lines1.jpg')
    # image = mpimg.imread('test_images/test4.jpg')
    image = mpimg.imread('test_images/test6.jpg')
    # np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

    # Read in the saved objpoints and imgpoints
    with open('points_dist_pickle.p', 'rb') as handle:
        dist_pickle = pickle.load(handle)
    objpoints = dist_pickle["objpoints"]
    imgpoints = dist_pickle["imgpoints"]
    detect_lanes(image)
