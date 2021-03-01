// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef PROJECTION_H
#define PROJECTION_H

// System headers
//

// Library headers
//

// Project headers
//
#include "charuco.h"
#include "utils.h"
using namespace cv;
using namespace std;
namespace k4aviewer
{
/**
 * \brief show image with given name
 */
void show_image(string window_name, Mat image)
{
    if (image.rows > 1000)
    {
        namedWindow(window_name, WINDOW_NORMAL);
        resizeWindow(window_name, 1400, 1050);
        moveWindow(window_name, 0, 0);
    }
    imshow(window_name, image);
    (char)waitKey(30);
}

enum class ProjectionMode
{
    Outer_3D_raycast_homography,
    Outer_3D_homography,
    Inner_3D,
    Inner_2D_calculated_corners,
    Inner_2D_detected_corners
};

class Projection
{
public:
    Projection(string input_video_1, string input_video_2, linmath::mat4x4 c2c) :
        video_1(input_video_1),
        video_2(input_video_2),
        m_charuco_1(input_video_1),
        m_charuco_2(input_video_2)
    {
        linmath::mat4x4_dup(m_c2c, c2c);
        linmath::mat4x4_dup(m_c2c_mm, c2c);
        // convert translation into mm
        float units_per_meter = 1000.0f;
        for (int i = 0; i < 3; i++)
            m_c2c_mm[3][i] *= units_per_meter;

        int number_of_frames = 5;
        double pixel_error = 0, distance_error = 0;
        for (int i = 0; i < number_of_frames; i++)
        {
            bool result_1 = false, result_2 = false;
            result_1 = video_1.next_capture_image();
            result_2 = video_2.next_capture_image();
            if (!result_1 || !result_2)
            {
                i--;
                continue;
            }

            m_charuco_1.detect_charuco_pose(video_1.m_colorMat);
            m_charuco_2.detect_charuco_pose(video_2.m_colorMat);

            // charuco inner corners
            calculate_charuco_corners(m_charuco_1,
                                      m_calculated_corners_1,
                                      m_calculated_corners_3d_1,
                                      m_calculated_ids_1);
            calculate_charuco_corners(m_charuco_2,
                                      m_calculated_corners_2,
                                      m_calculated_corners_3d_2,
                                      m_calculated_ids_2);
            // board corners
            calculate_charuco_board_corners(m_charuco_1, m_board_corners_1, m_board_corners_3d_1, m_calculated_ids_1);
            calculate_charuco_board_corners(m_charuco_2, m_board_corners_2, m_board_corners_3d_2, m_calculated_ids_2);

            Mat overlay_image;
            double alpha = 0.8;
            double beta_value;
            beta_value = (1.0 - alpha);

            ProjectionMode mode = ProjectionMode::Outer_3D_raycast_homography;
            switch (mode)
            {
            case k4aviewer::ProjectionMode::Outer_3D_raycast_homography:
                // 3d projection of board corners
                project_3d_to_3d(m_charuco_2,
                                 m_board_corners_3d_2,
                                 m_projected_board_corners_3d,
                                 m_projected_board_corners_from_3d);
                raycast(video_1.m_calibration,
                        video_1.m_depth_image,
                        video_1.m_depthMat,
                        m_projected_board_corners_from_3d,
                        m_raycast_corners_1);
                raycast(video_2.m_calibration,
                        video_2.m_depth_image,
                        video_2.m_depthMat,
                        m_board_corners_2,
                        m_raycast_corners_2);
                // extract board by homography
                homography(m_raycast_corners_1, video_1.m_colorMat, m_warpMat_1);
                homography(m_board_corners_2, video_2.m_colorMat, m_warpMat_2);
                // overlay
                addWeighted(m_warpMat_1, alpha, m_warpMat_2, beta_value, 0.0, overlay_image);
                show_image("1", m_warpMat_1);
                show_image("2", m_warpMat_2);
                show_image("overlay", overlay_image);
                break;

            case k4aviewer::ProjectionMode::Outer_3D_homography:
                // 3d projection of board corners
                project_3d_to_3d(m_charuco_2,
                                 m_board_corners_3d_2,
                                 m_projected_board_corners_3d,
                                 m_projected_board_corners_from_3d);
                // extract board by homography
                homography(m_projected_board_corners_from_3d, video_1.m_colorMat, m_warpMat_1);
                homography(m_board_corners_2, video_2.m_colorMat, m_warpMat_2);
                // overlay
                addWeighted(m_warpMat_1, alpha, m_warpMat_2, beta_value, 0.0, overlay_image);
                show_image("1", m_warpMat_1);
                show_image("2", m_warpMat_2);
                show_image("overlay", overlay_image);
                break;

            case k4aviewer::ProjectionMode::Inner_3D:
                // 3d projection of calculated corners
                project_3d_to_3d(m_charuco_2,
                                 m_calculated_corners_3d_2,
                                 m_projected_corners_3d,
                                 m_projected_corners_from_3d);
                // 3d error in mm
                distance_error += distance_error_by_frame_per_point(m_calculated_corners_3d_1,
                                                                    m_calculated_ids_1,
                                                                    m_projected_corners_3d,
                                                                    m_calculated_ids_2);
                // 2d error in pixels
                pixel_error += pixel_error_by_frame_per_point(m_calculated_corners_1,
                                                              m_calculated_ids_1,
                                                              m_projected_corners_from_3d,
                                                              m_calculated_ids_2);
                draw(m_calculated_corners_1, m_calculated_ids_1, m_projected_corners_from_3d, m_calculated_ids_2);
                break;

            case k4aviewer::ProjectionMode::Inner_2D_calculated_corners:
                // 2d projection of calculated corners (using kinect sdk)
                project_2d_to_2d(video_2.m_calibration,
                                 video_2.m_depth_image,
                                 video_2.m_depthMat,
                                 m_calculated_corners_2,
                                 m_projected_corners);
                // 2d error in pixels
                pixel_error += pixel_error_by_frame_per_point(m_calculated_corners_1,
                                                              m_calculated_ids_1,
                                                              m_projected_corners,
                                                              m_calculated_ids_2);
                draw(m_calculated_corners_1, m_calculated_ids_1, m_projected_corners, m_calculated_ids_2);
                break;

            case k4aviewer::ProjectionMode::Inner_2D_detected_corners:
                // 2d projection of detected corners (using kinect sdk)
                project_2d_to_2d(video_2.m_calibration,
                                 video_2.m_depth_image,
                                 video_2.m_depthMat,
                                 m_charuco_2.m_detected_corners,
                                 m_projected_corners);
                // 2d error in pixels
                pixel_error += pixel_error_by_frame_per_point(m_charuco_1.m_detected_corners,
                                                              m_charuco_1.m_detected_ids,
                                                              m_projected_corners,
                                                              m_charuco_2.m_detected_ids);
                draw(m_charuco_1.m_detected_corners,
                     m_charuco_1.m_detected_ids,
                     m_projected_corners,
                     m_charuco_2.m_detected_ids);
                break;

            default:
                throw std::runtime_error("Selected an unsupported projection type!");
                break;
            }

            video_1.release_images();
            video_2.release_images();
        }
        double average_pixel_error = pixel_error / number_of_frames,
               average_distance_error = distance_error / number_of_frames;
        average_pixel_error;
        average_distance_error; 
        video_1.close_playback();
        video_2.close_playback();
    }

    void homography(vector<Point2f> &corners, Mat &color_image, Mat &warped_image) {
        vector<Point2f> new_corners;
        new_corners.push_back(Point2f(640, 480));
        new_corners.push_back(Point2f(640, 0));
        new_corners.push_back(Point2f(0, 0));
        new_corners.push_back(Point2f(0, 480));
        Mat H = findHomography(corners, new_corners);
        warpPerspective(color_image, warped_image, H, Size(640, 480));
    }

    void project_3d_to_3d(Charuco &charuco,
                          vector<Point3f> &old_corners,
                          vector<Point3f> &new_corners_3d,
                          vector<Point2f> &new_corners)
    {
        // 3d projection by camera to camera matrix
        new_corners_3d.clear();
        for (int i = 0; i < old_corners.size(); i++)
        {
            linmath::vec4 point_4d{ old_corners[i].x, old_corners[i].y, old_corners[i].z, 1.0f }, projected_point_4d;
            linmath::mat4x4_mul_vec4(projected_point_4d, m_c2c, point_4d);
            
            new_corners_3d.push_back(
                Point3f(projected_point_4d[0], projected_point_4d[1], projected_point_4d[2]));
        }

        // projection to 2d pixel coordinates
        vector<Point3f> converted;
        converted = convert_for_2d(charuco.m_rvec, new_corners_3d);
        projectPoints(converted,
                      charuco.m_rvec,
                      charuco.m_tvec,
                      charuco.m_cameraMatrix,
                      charuco.m_distCoeffs,
                      new_corners);
    }

    void raycast(k4a_calibration_t &calibration,
                 k4a_image_t &depth_image,
                 Mat &depthMat,
                 vector<Point2f> &old_corners,
                 vector<Point2f> &new_corners)
    {
        project_2d_to_2d(calibration, depth_image, depthMat, old_corners, new_corners, true);
    }

    void project_2d_to_2d(k4a_calibration_t &calibration,
                          k4a_image_t &depth_image,
                          Mat &depthMat,
                          vector<Point2f> &old_corners,
                          vector<Point2f> &new_corners,
                          bool raycast = false)
    {
        new_corners.clear();
        for (int i = 0; i < old_corners.size(); i++)
        {
            k4a_float2_t point_2d_depth, point_2d, point_2d_projected;
            k4a_float3_t point_3d, point_3d_depth, point_3d_projected;
            int valid;

            // Detected 2d Charuco corner point
            point_2d.xy.x = static_cast<float>(old_corners[i].x);
            point_2d.xy.y = static_cast<float>(old_corners[i].y);

            // Get depth of the 2d point
            k4a_calibration_color_2d_to_depth_2d(&calibration,
                                                 &point_2d,
                                                 depth_image,
                                                 &point_2d_depth,
                                                 &valid);
            int idx_col = (int)round(point_2d_depth.xy.x);
            int idx_row = (int)round(point_2d_depth.xy.y);
            float point_2d_depth_value = depthMat.at<uint16_t>(idx_row, idx_col);

            if (raycast)
            {
                // Project 2d point in color to 3d point in depth
                k4a_calibration_2d_to_3d(&calibration,
                                         &point_2d,
                                         point_2d_depth_value,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         K4A_CALIBRATION_TYPE_DEPTH,
                                         &point_3d_depth,
                                         &valid);

                // Project 3d point in depth to 3d point in color
                k4a_calibration_3d_to_3d(&calibration,
                                         &point_3d_depth,
                                         K4A_CALIBRATION_TYPE_DEPTH,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         &point_3d);

                // Project new 3d point in color to 2d point in color
                k4a_calibration_3d_to_2d(&calibration,
                                         &point_3d,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         &point_2d_projected,
                                         &valid);
            }
            else
            {
                // Project 2d point to 3d point
                k4a_calibration_2d_to_3d(&calibration,
                                         &point_2d,
                                         point_2d_depth_value,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         &point_3d,
                                         &valid);

                // Multiply 3d point with camera2camera matrix
                linmath::vec4 point_4d{ point_3d.xyz.x, point_3d.xyz.y, point_3d.xyz.z, 1.0f }, projected_point_4d;
                linmath::mat4x4_mul_vec4(projected_point_4d, m_c2c_mm, point_4d);

                // Remove last dimension
                point_3d_projected.xyz.x = static_cast<float>(projected_point_4d[0]);
                point_3d_projected.xyz.y = static_cast<float>(projected_point_4d[1]);
                point_3d_projected.xyz.z = static_cast<float>(projected_point_4d[2]);

                // Project new 3d point back to 2d
                k4a_calibration_3d_to_2d(&calibration,
                                         &point_3d_projected,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         &point_2d_projected,
                                         &valid);
            }

            new_corners.push_back(Point2f(point_2d_projected.xy.x, point_2d_projected.xy.y));
        }
    }

    /**
    * \brief calculate corner points by getting Horizontal and Vertical chess square vectors
    */
    vector<Point3f> get_corners_in_camera_world(double side, Vec3d rvec, Vec3d tvec, bool board_corners = false)
    {
        // compute rot_mat
        Mat rot_mat;
        Rodrigues(rvec, rot_mat);

        // transpose of rot_mat for easy columns extraction
        Mat rot_mat_t = rot_mat.t();

        // the two E-O and F-O vectors
        double *tmp = rot_mat_t.ptr<double>(0);
        Point3f cam_world_E((float)(tmp[0] * side), (float)(tmp[1] * side), (float)(tmp[2] * side));

        tmp = rot_mat_t.ptr<double>(1);
        Point3f cam_world_F((float)(tmp[0] * side), (float)(tmp[1] * side), (float)(tmp[2] * side));

        // convert tvec to point
        Point3f tvec_3f((float)tvec[0], (float)tvec[1], (float)tvec[2]);
        vector<Point3f> result;
        if (board_corners)
        {
            // 4 corners of the board
            vector<Point3f> ret(4, tvec_3f);
            ret[1] += 4 * cam_world_E;
            ret[2] += 4 * cam_world_E + 3 * cam_world_F;
            ret[3] += 3 * cam_world_F;
            return ret;
        }
        else
        {
            // 6 inner corners of charuco
            vector<Point3f> ret(6, tvec_3f);
            ret[0] += cam_world_E + cam_world_F;
            ret[1] += 2 * cam_world_E + cam_world_F;
            ret[2] += 3 * cam_world_E + cam_world_F;
            ret[3] += cam_world_E + 2 * cam_world_F;
            ret[4] += 2 * cam_world_E + 2 * cam_world_F;
            ret[5] += 3 * cam_world_E + 2 * cam_world_F;
            return ret;
        }
    }

    vector<Point3f> convert_for_2d(Vec3d &rvec, vector<Point3f> original)
    {
        vector<Point3f> result; 
        for (int i = 0; i < original.size(); i++)
        {
            Vec3d tvec(original[i].x, original[i].y, original[i].z);
            Mat R;
            Rodrigues(rvec, R);
            R = R.t();
            Mat new_tvec = -R * tvec;
            double *tmp = new_tvec.ptr<double>(0);
            result.push_back(Point3f((float)tmp[0] * 1000, (float)tmp[1] * 1000, (float)tmp[2] * 1000));
        }
        return result;
    }

    void calculate_charuco_board_corners(Charuco &charuco,
                                         vector<Point2f> &calculated_corners,
                                         vector<Point3f> &calculated_corners_3d,
                                         vector<int> &calculated_ids)
    {
        calculate_charuco_corners(charuco, calculated_corners, calculated_corners_3d, calculated_ids, true);
    }
    void calculate_charuco_corners(Charuco &charuco,
                                   vector<Point2f> &calculated_corners,
                                   vector<Point3f> &calculated_corners_3d,
                                   vector<int> &calculated_ids,
                                   bool is_board_corners = false)
    {
        calculated_corners_3d = get_corners_in_camera_world(0.053, charuco.m_rvec, charuco.m_tvec, is_board_corners);
        vector<Point3f> converted;
        converted = convert_for_2d(charuco.m_rvec, calculated_corners_3d);
        projectPoints(converted,
                      charuco.m_rvec,
                      charuco.m_tvec,
                      charuco.m_cameraMatrix,
                      charuco.m_distCoeffs,
                      calculated_corners);
        
        if (!is_board_corners)
        {
            calculated_ids.resize(calculated_corners_3d.size());
            generate(calculated_ids.begin(), calculated_ids.end(), [n = 0]() mutable { return n++; });
        }
    }

    void draw(vector<Point2f> &corners_1, vector<int> &ids_1, vector<Point2f> &corners_2, vector<int> &ids_2)
    {
        aruco::drawDetectedCornersCharuco(video_1.m_colorMat, corners_2, ids_2, Scalar(0, 0, 255)); // aligned corners
                                                                                                    // in camera6
        aruco::drawDetectedCornersCharuco(video_1.m_colorMat, corners_1, ids_1, Scalar(0, 255, 0)); // reference corners
                                                                                                    // in camera3
        aruco::drawAxis(video_1.m_colorMat,
                        m_charuco_1.m_cameraMatrix,
                        m_charuco_1.m_distCoeffs,
                        m_charuco_1.m_rvec,
                        m_charuco_1.m_tvec,
                        0.1f);
        Mat croppedFrame = video_1.m_colorMat(Rect(video_1.m_colorMat.cols / 2,
                                                   video_1.m_colorMat.rows / 4,
                                                   video_1.m_colorMat.cols / 3,
                                                   video_1.m_colorMat.rows / 3));
        show_image("Projected", croppedFrame);
        //show_image("Projected", video_1.m_colorMat);
    }

    double distance_error_by_frame_per_point(vector<Point3f> corners_3d_1,
                                             vector<int> ids_1,
                                             vector<Point3f> corners_3d_2,
                                             vector<int> ids_2)
    {
        return error_by_frame_per_point(
            vector<Point2f>(), corners_3d_1, ids_1, vector<Point2f>(), corners_3d_2, ids_2, true);
    }
    double pixel_error_by_frame_per_point(vector<Point2f> corners_1,
                                          vector<int> ids_1,
                                          vector<Point2f> corners_2,
                                          vector<int> ids_2)
    {
        return error_by_frame_per_point(corners_1, vector<Point3f>(), ids_1, corners_2, vector<Point3f>(), ids_2);
    }

    double error_by_frame_per_point(vector<Point2f> corners_1,
                                    vector<Point3f> corners_3d_1,
                                    vector<int> ids_1,
                                    vector<Point2f> corners_2,
                                    vector<Point3f> corners_3d_2,
                                    vector<int> ids_2,
                                    bool error_in_3d = false)
    {
        int idx_1 = 0, idx_2 = 0;
        double frame_error = 0;
        int number_of_corners = 0;
        while (idx_1 < ids_1.size() && idx_2 < ids_2.size())
        {
            if (ids_1[idx_1] == ids_2[idx_2])
            {
                double e;
                if (!error_in_3d)
                {
                    e = norm(corners_1[idx_1] - corners_2[idx_2]);
                    arrowedLine(video_1.m_colorMat, corners_1[idx_1], corners_2[idx_2], Scalar(255, 0, 255), 2, LINE_AA, 0, 0.3);
                }
                else
                    e = norm(corners_3d_1[idx_1] - corners_3d_2[idx_2]) * 1000.0; // error in mm
                frame_error += e;
                number_of_corners++;
                idx_1++;
                idx_2++;
            }
            else if (ids_1[idx_1] < ids_2[idx_2])
                idx_1++;
            else
                idx_2++;
        }
        return frame_error / number_of_corners;
    }

    vector<Point2f> m_board_corners_1, m_board_corners_2;
    vector<Point2f> m_raycast_corners_1, m_raycast_corners_2;
    vector<Point2f> m_calculated_corners_1, m_calculated_corners_2;
    vector<Point3f> m_calculated_corners_3d_1, m_calculated_corners_3d_2;
    vector<Point3f> m_board_corners_3d_1, m_board_corners_3d_2;
    vector<int> m_calculated_ids_1, m_calculated_ids_2;
    vector<Point2f> m_projected_corners, m_projected_corners_from_3d, m_projected_board_corners_from_3d;
    vector<Point3f> m_projected_corners_3d, m_projected_board_corners_3d;


    linmath::mat4x4 m_c2c, m_c2c_mm;
    Charuco m_charuco_1, m_charuco_2;

    Video video_1, video_2;

    Mat m_warpMat_1, m_warpMat_2;
};
} // namespace k4aviewer

#endif
