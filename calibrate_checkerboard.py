# -*- coding: utf-8 -*-
"""
Line-by-line commented version of the checkerboard calibration script.
This script estimates camera intrinsics (fx, fy, cx, cy) and distortion coefficients
from multiple images (or sampled video frames) of a printed checkerboard using OpenCV.
"""

import argparse          # For parsing command-line arguments
import glob              # For expanding file patterns like "data/*.jpg"
import os                # For filesystem operations (creating folders, joining paths)
import cv2               # OpenCV: image I/O, corner detection, calibration utilities
import numpy as np       # Numerical computations and array handling


def collect_images_from_video(video_path, out_dir, max_frames=40, skip=10):
    """
    Extracts frames from a video at a fixed interval and saves them as images.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    out_dir : str
        Directory where sampled frames will be saved (created if not present).
    max_frames : int, optional
        Maximum number of frames to extract (default: 40).
    skip : int, optional
        Keep 1 frame out of every `skip` frames (default: 10). For example,
        if skip=10 then frames #0, #10, #20, ... are saved.

    Returns
    -------
    list[str]
        List of file paths to the saved frame images.

    Raises
    ------
    RuntimeError
        If the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)                     # Open the video file
    if not cap.isOpened():                                 # Verify it opened successfully
        raise RuntimeError(f"Could not open video: {video_path}")
    os.makedirs(out_dir, exist_ok=True)                    # Ensure output folder exists
    saved = []                                             # Accumulate saved frame paths
    idx = 0                                                # Counter for saved frames
    frame_id = 0                                           # Counter for all frames read
    while True:                                            # Read until video ends
        ok, frame = cap.read()                             # Grab next frame
        if not ok:                                         # Stop if no more frames
            break
        if frame_id % skip == 0:                           # Sample every `skip`-th frame
            out = os.path.join(out_dir, f"frame_{idx:04d}.jpg")  # Build output filename
            cv2.imwrite(out, frame)                        # Save the current frame as JPEG
            saved.append(out)                              # Record saved path
            idx += 1                                       # Increment saved-frame index
            if idx >= max_frames:                          # Stop if reached max_frames
                break
        frame_id += 1                                      # Move to next frame index
    cap.release()                                          # Release the video handle
    return saved                                           # Return list of saved images


def main():
    """
    Parses CLI arguments, loads checkerboard images (or samples video frames),
    detects inner corners, and runs OpenCV calibration to recover intrinsics and distortion.

    Workflow:
    1) Gather image paths via --images_glob OR sample frames via --video.
    2) For each image: read -> grayscale -> find chessboard -> refine corners.
    3) Run cv2.calibrateCamera to estimate K and distortion coefficients.
    4) Report fx, fy, cx, cy and mean reprojection error. Optionally save undistorted previews.
    """
    parser = argparse.ArgumentParser(description="Checkerboard camera calibration (OpenCV)")  # Build CLI parser
    parser.add_argument("--images_glob", type=str, default="", help="Glob for calibration images, e.g., 'data/*.jpg'")  # Image pattern
    parser.add_argument("--video", type=str, default="", help="Video file to sample frames from")                        # Video path
    parser.add_argument("--video_out_dir", type=str, default="calib_frames", help="Where to save frames from video")    # Extracted frames folder
    parser.add_argument("--video_max_frames", type=int, default=40, help="Max frames to sample from video")             # Max frames to sample
    parser.add_argument("--video_skip", type=int, default=10, help="Take 1 every N frames")                             # Sampling stride
    parser.add_argument("--nx", type=int, default=9, help="Inner corners across (columns)")                             # Inner corners along width
    parser.add_argument("--ny", type=int, default=6, help="Inner corners down (rows)")                                  # Inner corners along height
    parser.add_argument("--square_size_mm", type=float, default=25.0, help="Physical square size in mm (for scale)")    # Square size in mm
    parser.add_argument("--show", action="store_true", help="Show corner detections as a preview")                      # Option to visualize corners
    parser.add_argument("--save_undistorted", action="store_true", help="Save undistorted previews")                    # Option to save undistorted images
    parser.add_argument("--save_dir", type=str, default="calib_output", help="Output directory for results")            # Output folder for report/images
    args = parser.parse_args()                                                                                          # Parse arguments from CLI

    img_paths = []                                                                                                      # Will hold image file paths
    if args.images_glob:                                                                                                # Case 1: use glob pattern
        img_paths = sorted(glob.glob(args.images_glob))                                                                 # Expand and sort paths
    elif args.video:                                                                                                    # Case 2: sample from video
        img_paths = collect_images_from_video(args.video, args.video_out_dir,                                           # Extract frames...
                                              max_frames=args.video_max_frames, skip=args.video_skip)                   # ...with sampling parameters
    else:
        raise SystemExit("Provide --images_glob or --video")                                                            # Require at least one source

    if not img_paths:                                                                                                   # Guard: no images found
        raise SystemExit("No images found.")

    os.makedirs(args.save_dir, exist_ok=True)                                                                           # Ensure output directory exists

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)                                          # Termination criteria for cornerSubPix

    nx, ny = args.nx, args.ny                                                                                           # Read pattern size (inner corners)
    square_size = args.square_size_mm                                                                                   # Physical square size (mm)
    objp = np.zeros((ny*nx, 3), np.float32)                                                                             # Allocate 3D object points (Z=0 plane)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2) * square_size                                                     # Create (x,y) grid scaled by square_size

    objpoints = []                                                                                                      # Accumulates 3D points for each image
    imgpoints = []                                                                                                      # Accumulates 2D detected corners for each image

    img_size = None                                                                                                     # Will store image size (width, height)
    kept = 0                                                                                                            # Count of images with successful detections

    for p in img_paths:                                                                                                 # Iterate over all image paths
        img = cv2.imread(p)                                                                                             # Read image from disk (may fail on non-ASCII paths on Windows)
        if img is None:                                                                                                 # Skip if failed to read
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                                                    # Convert to grayscale for corner detection
        if img_size is None:                                                                                            # Record image size once (w, h)
            img_size = gray.shape[::-1]                                                                                 # shape[::-1] -> (width, height)

        # Find inner chessboard corners (pattern size is (nx, ny)).
        ret, corners = cv2.findChessboardCorners(                                                                       # Attempt corner detection
            gray, (nx, ny),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE                                           # Helpful pre-processing flags
        )
        if ret:                                                                                                         # If detection succeeded
            corners = cv2.cornerSubPix(                                                                                 # Refine corner locations to sub-pixel accuracy
                gray, corners, (11,11), (-1,-1), criteria
            )
            objpoints.append(objp.copy())                                                                               # Append corresponding 3D points
            imgpoints.append(corners)                                                                                   # Append detected/refined 2D points
            kept += 1                                                                                                   # Count valid detection

            if args.show:                                                                                               # Optional visualization
                vis = cv2.drawChessboardCorners(img.copy(), (nx, ny), corners, ret)                                     # Draw detected corners
                cv2.imshow("corners", vis)                                                                              # Show window
                cv2.waitKey(300)                                                                                        # Brief pause (300 ms)
        else:                                                                                                           # If detection failed
            if args.show:                                                                                               # Optionally show original image
                cv2.imshow("corners", img)
                cv2.waitKey(200)

    if args.show:                                                                                                       # If windows were opened,
        cv2.destroyAllWindows()                                                                                         # close all OpenCV windows

    if kept < 5:                                                                                                        # Require sufficient views for stability
        raise SystemExit(f"Not enough valid detections: got {kept} frames; need >=5.")

    flags = cv2.CALIB_RATIONAL_MODEL                                                                                    # Use rational distortion model (k1..k6 + p1,p2)
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(                                                                   # Perform camera calibration
        objpoints, imgpoints, img_size, None, None, flags=flags
    )

    fx, fy = K[0,0], K[1,1]                                                                                             # Extract focal lengths in pixels
    cx, cy = K[0,2], K[1,2]                                                                                             # Extract principal point coordinates

    mean_error = 0                                                                                                      # Accumulate mean reprojection error
    for i in range(len(objpoints)):                                                                                     # For each view (image)
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)                                    # Reproject 3D object points
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)                                       # Compute average L2 pixel error
        mean_error += error                                                                                              # Sum errors
    mean_error /= len(objpoints)                                                                                        # Average over all views

    report = os.path.join(args.save_dir, "calibration_report.txt")                                                      # Path to save the text report
    with open(report, "w") as f:                                                                                        # Write results to file
        f.write("=== Camera Calibration Report ===\n")
        f.write(f"Image size: {img_size}\n")
        f.write(f"Frames used: {kept} / {len(img_paths)}\n")
        f.write(f"RMS (cv2.calibrateCamera): {ret:.6f}\n")                                                               # RMS returned by OpenCV
        f.write(f"Mean reprojection error (px): {mean_error:.6f}\n\n")                                                   # Our own mean reprojection error
        f.write("Intrinsic matrix K:\n")
        f.write(str(K) + "\n\n")                                                                                         # K printed as a 3x3 matrix
        f.write("Distortion coefficients (k1,k2,p1,p2,k3,k4,k5,k6):\n")
        f.write(str(dist.ravel()) + "\n\n")                                                                              # Distortion as a flat array
        f.write(f"fx: {fx:.6f} px\n")                                                                                    # Focal length x
        f.write(f"fy: {fy:.6f} px\n")                                                                                    # Focal length y
        f.write(f"cx: {cx:.6f} px\n")                                                                                    # Principal point x
        f.write(f"cy: {cy:.6f} px\n")                                                                                    # Principal point y

    print("=== Calibration result ===")                                                                                 # Also print a concise summary to stdout
    print(f"Image size: {img_size}")
    print(f"Frames used: {kept} / {len(img_paths)}")
    print(f"RMS: {ret:.6f}")
    print(f"Mean reprojection error (px): {mean_error:.6f}")
    print("K =\n", K)
    print("dist =\n", dist.ravel())
    print(f"fx, fy = {fx:.3f}, {fy:.3f}  (pixels)")
    print(f"cx, cy = {cx:.3f}, {cy:.3f}  (pixels)")
    print(f"Report saved to: {report}")

    if args.save_undistorted:                                                                                           # Optionally save undistorted previews
        und_dir = os.path.join(args.save_dir, "undistorted")                                                            # Output folder for previews
        os.makedirs(und_dir, exist_ok=True)                                                                             # Ensure it exists
        for i, p in enumerate(img_paths[:10]):                                                                          # Limit to first 10 images
            img = cv2.imread(p)                                                                                          # Read original image
            if img is None:                                                                                              # Guard: skip unreadable files
                continue
            und = cv2.undistort(img, K, dist)                                                                            # Apply undistortion using estimated params
            cv2.imwrite(os.path.join(und_dir, f"und_{i:03d}.jpg"), und)                                                  # Save undistorted image

if __name__ == "__main__":                                                                                               # Standard Python entry point
    main()                                                                                                               # Run the main() routine