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
void show_image(string window_name, Mat image, int x_pos = 0, int y_pos = 0)
{
    namedWindow(window_name, WINDOW_NORMAL);
    if (image.rows > 1000)
    {
        resizeWindow(window_name, 1400, 1050);
    }
    else
    {
        resizeWindow(window_name, 400, 400);
    }
    moveWindow(window_name, x_pos, y_pos);
    imshow(window_name, image);
    (char)waitKey(30);
}

enum class ProjectionMode
{
    Find_plane,
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

            Mat overlay_image, for_overlay_1, for_overlay_2;
            double alpha = 0.8;
            double beta_value;
            beta_value = (1.0 - alpha);
            int block_size = 10, offset = 70;
            int block_width, block_height;
            Mat disp;
            Point match_loc;
            
            ProjectionMode mode = ProjectionMode::Outer_3D_raycast_homography;
            switch (mode)
            {
            case k4aviewer::ProjectionMode::Find_plane:

                find_plane(video_2.m_calibration, video_2.m_depthMat, m_planes_2, m_planes_3d_2, block_size);
                for (int j = 0; j < m_planes_3d_2.size(); j++)
                {
                    color_corners_of_depth_planes(video_2.m_calibration, m_planes_3d_2[j], m_temp_corners_2);
                    if (!project_planes(video_1.m_calibration,
                                        video_1.m_depth_image,
                                        video_1.m_depthMat,
                                        video_2.m_calibration,
                                        m_planes_3d_2[j],
                                        m_temp_corners_1))
                        continue;

                    if (!sanity_check(m_temp_corners_1,
                                      video_1.m_calibration,
                                      m_temp_corners_2,
                                      video_2.m_calibration,
                                      block_width,
                                      block_height))
                        continue;

                    match_loc = template_matching_pipeline(m_temp_corners_1, m_temp_corners_2, block_width, block_height, offset);

                    create_hom_corners(m_hom_corners_2,
                                       (float)block_width,
                                       (float)block_height,
                                       (float)offset,
                                       (float)offset);
                    create_hom_corners(m_hom_corners_1,
                                       (float)block_width,
                                       (float)block_height,
                                       (float)match_loc.x,
                                       (float)match_loc.y);

                    while (m_calculated_ids_1.size() > 4)
                    {
                        m_calculated_ids_1.pop_back();
                        m_calculated_ids_2.pop_back();
                    }
                    // 2d error in pixels
                    pixel_error += pixel_error_by_frame_per_point(m_hom_corners_1,
                                                                  m_calculated_ids_1,
                                                                  m_hom_corners_2,
                                                                  m_calculated_ids_2,
                                                                  m_warpMat_1);

                    show_image("error", m_warpMat_1, 810, 440);

                    /*
                    Mat disp1, disp2;
                    video_1.m_colorMat.copyTo(disp1);
                    video_2.m_colorMat.copyTo(disp2);

                    rectangle(disp1, m_temp_corners_1[0], m_temp_corners_1[2], Scalar(0, 0, 255), 2, 8, 0);
                    rectangle(disp2, m_temp_corners_2[0], m_temp_corners_2[2], Scalar(0, 0, 255), 2, 8, 0);

                    show_image("1", disp1);
                    show_image("2", disp2, 1000);*/
                }
                break;
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
                homography(m_raycast_corners_1, video_1.m_colorMat, for_overlay_1);
                homography(m_raycast_corners_2, video_2.m_colorMat, for_overlay_2);
                // overlay
                addWeighted(for_overlay_1, alpha, for_overlay_2, beta_value, 0.0, overlay_image);
                show_image("overlay", overlay_image, 405, 440);

                block_width = 640;
                block_height = 480;
                offset = 200;
                match_loc = template_matching_pipeline(m_raycast_corners_1, m_raycast_corners_2, block_width, block_height, 200);

                create_hom_corners(m_hom_corners_2, (float)block_width, (float)block_height, (float)offset, (float)offset);
                create_hom_corners(m_hom_corners_1, (float)block_width, (float)block_height, (float)match_loc.x, (float)match_loc.y);

                while (m_calculated_ids_1.size() > 4)
                {
                    m_calculated_ids_1.pop_back();
                    m_calculated_ids_2.pop_back();
                }
                // 2d error in pixels
                pixel_error += pixel_error_by_frame_per_point(m_hom_corners_1,
                                                              m_calculated_ids_1,
                                                              m_hom_corners_2,
                                                              m_calculated_ids_2,
                                                              m_warpMat_1);

                show_image("error", m_warpMat_1, 810, 440);
                break;

            case k4aviewer::ProjectionMode::Outer_3D_homography:
                // 3d projection of board corners
                project_3d_to_3d(m_charuco_2,
                                 m_board_corners_3d_2,
                                 m_projected_board_corners_3d,
                                 m_projected_board_corners_from_3d);

                // extract board by homography
                homography(m_projected_board_corners_from_3d, video_1.m_colorMat, for_overlay_1);
                homography(m_board_corners_2, video_2.m_colorMat, for_overlay_2);
                // overlay
                addWeighted(for_overlay_1, alpha, for_overlay_2, beta_value, 0.0, overlay_image);
                show_image("overlay", overlay_image, 405, 440);

                block_width = 640;
                block_height = 480;
                offset = 200;
                match_loc = template_matching_pipeline(m_projected_board_corners_from_3d,
                                                       m_board_corners_2,
                                                       block_width,
                                                       block_height,
                                                       200);

                create_hom_corners(m_hom_corners_2,
                                   (float)block_width,
                                   (float)block_height,
                                   (float)offset,
                                   (float)offset);
                create_hom_corners(m_hom_corners_1,
                                   (float)block_width,
                                   (float)block_height,
                                   (float)match_loc.x,
                                   (float)match_loc.y);

                while (m_calculated_ids_1.size() > 4)
                {
                    m_calculated_ids_1.pop_back();
                    m_calculated_ids_2.pop_back();
                }
                // 2d error in pixels
                pixel_error += pixel_error_by_frame_per_point(m_hom_corners_1,
                                                              m_calculated_ids_1,
                                                              m_hom_corners_2,
                                                              m_calculated_ids_2,
                                                              m_warpMat_1);

                show_image("error", m_warpMat_1, 810, 440);
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
                                                              m_calculated_ids_2,
                                                              video_1.m_colorMat);
                draw(m_calculated_corners_1, m_calculated_ids_1, m_projected_corners_from_3d, m_calculated_ids_2);
                break;

            case k4aviewer::ProjectionMode::Inner_2D_calculated_corners:
                // 2d projection of calculated corners (using kinect sdk)
                project_2d_to_2d(video_1.m_calibration,
                                 video_2.m_calibration,
                                 video_2.m_depth_image,
                                 video_2.m_depthMat,
                                 m_calculated_corners_2,
                                 m_projected_corners);
                // 2d error in pixels
                pixel_error += pixel_error_by_frame_per_point(m_calculated_corners_1,
                                                              m_calculated_ids_1,
                                                              m_projected_corners,
                                                              m_calculated_ids_2,
                                                              video_1.m_colorMat);
                draw(m_calculated_corners_1, m_calculated_ids_1, m_projected_corners, m_calculated_ids_2);
                break;

            case k4aviewer::ProjectionMode::Inner_2D_detected_corners:
                // 2d projection of detected corners (using kinect sdk)
                project_2d_to_2d(video_1.m_calibration,
                                 video_2.m_calibration,
                                 video_2.m_depth_image,
                                 video_2.m_depthMat,
                                 m_charuco_2.m_detected_corners,
                                 m_projected_corners);
                // 2d error in pixels
                pixel_error += pixel_error_by_frame_per_point(m_charuco_1.m_detected_corners,
                                                              m_charuco_1.m_detected_ids,
                                                              m_projected_corners,
                                                              m_charuco_2.m_detected_ids,
                                                              video_1.m_colorMat);
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

    void create_hom_corners(vector<Point2f> &hom_corners, float width, float height, float offset_x, float offset_y)
    {
        hom_corners.clear();
        hom_corners.push_back(Point2f(width + offset_x, height + offset_y));
        hom_corners.push_back(Point2f(width + offset_x, 0 + offset_y));
        hom_corners.push_back(Point2f(0 + offset_x, 0 + offset_y));
        hom_corners.push_back(Point2f(0 + offset_x, height + offset_y));
    }

    Point template_matching_pipeline(vector<Point2f> &full_image_corners, vector<Point2f> &template_corners, int block_width, int block_height, int offset) {
        homography(template_corners, video_2.m_colorMat, m_warpMat_2, block_width, block_height);
        homography(full_image_corners, video_1.m_colorMat, m_warpMat_1, block_width, block_height, offset);

        Mat img_match;
        m_warpMat_1.copyTo(img_match);
        Point match_loc;
        match_loc = template_matching(m_warpMat_1, m_warpMat_2);

        rectangle(img_match,
                  match_loc,
                  Point(match_loc.x + block_width, match_loc.y + block_height),
                  Scalar(0, 0, 255),
                  3,
                  8,
                  0);
        show_image("Template matching", img_match, 405);

        Mat img_search;
        m_warpMat_1.copyTo(img_search);
        rectangle(img_search,
                  Point(offset, offset),
                  Point(offset + block_width, offset + block_height),
                  Scalar(0, 0, 255),
                  3,
                  8,
                  0);
        show_image("search image (projected)", img_search, 810);
        show_image("template", m_warpMat_2, 0, 440);

        Mat disp;
        homography(template_corners, video_2.m_colorMat, disp, block_width, block_height, offset);
        rectangle(disp,
                  Point(offset, offset),
                  Point(offset + block_width, offset + block_height),
                  Scalar(0, 0, 255),
                  3,
                  8,
                  0);
        show_image("block original place", disp, 0);
        return match_loc;
    }

    bool limit_check(vector<Point2f> &corners, int width_limit, int height_limit) {
        for (int i = 0; i < corners.size(); i++)
        {
            if (corners[i].x < 0 || corners[i].x >= width_limit)
                return false;
            if (corners[i].y < 0 || corners[i].y >= height_limit)
                return false;
        }
        return true;
    }
    void get_dims(vector<Point2f> &corners, int width_limit, int height_limit, int &block_width, int &block_height)
    {
        float min_width = (float)width_limit, min_height = (float)height_limit;
        float max_width = -1, max_height = -1;
        for (int i = 0; i < corners.size(); i++)
        {
            if (min_width > corners[i].x)
                min_width = corners[i].x;
            if (max_width < corners[i].x)
                max_width = corners[i].x;
            if (min_height > corners[i].y)
                min_height = corners[i].y;
            if (max_height < corners[i].y)
                max_height = corners[i].y;
        }
        block_width = (int)round(max_width - min_width);
        block_height = (int)round(max_height - min_height);
    }

    bool sanity_check(vector<Point2f> &corners_1,
                      k4a_calibration_t &calibration_1,
                      vector<Point2f> &corners_2,
                      k4a_calibration_t &calibration_2,
                      int &block_width,
                      int &block_height)
    {
        if (!limit_check(corners_1,
                         calibration_1.color_camera_calibration.resolution_width,
                         calibration_1.color_camera_calibration.resolution_height))
            return false;
        if (!limit_check(corners_2,
                         calibration_2.color_camera_calibration.resolution_width,
                         calibration_2.color_camera_calibration.resolution_height))
            return false;

        get_dims(corners_2,
                 calibration_2.color_camera_calibration.resolution_width,
                 calibration_2.color_camera_calibration.resolution_height,
                 block_width,
                 block_height); 

        return true;
    }

    Point2f template_matching(Mat &full_image, Mat &temp)
    {
        Mat result;
        int result_cols = full_image.cols - temp.cols + 1;
        int result_rows = full_image.rows - temp.rows + 1;
        result.create(result_rows, result_cols, CV_32FC1);
        
        int match_method = TM_SQDIFF; 
        matchTemplate(full_image, temp, result, match_method);
        normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());

        double minVal;
        double maxVal;
        Point min_loc;
        Point max_loc;
        Point match_loc;
        minMaxLoc(result, &minVal, &maxVal, &min_loc, &max_loc, Mat());
        if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED)
            match_loc = min_loc;
        else
            match_loc = max_loc;

        return match_loc;
    }
    bool plane_equation(Point3f p1, Point3f p2, Point3f p3, Point3f p)
    {
        float a1 = p2.x - p1.x;
        float b1 = p2.y - p1.y;
        float c1 = p2.z - p1.z;
        float a2 = p3.x - p1.x;
        float b2 = p3.y - p1.y;
        float c2 = p3.z - p1.z;
        float a = b1 * c2 - b2 * c1;
        float b = a2 * c1 - a1 * c2;
        float c = a1 * b2 - b1 * a2;
        float d = (-a * p1.x - b * p1.y - c * p1.z);

        // equation of plane is: a*x + b*y + c*z = 0 #
        float equation = a * p.x + b * p.y + c * p.z + d;
        
        // checking if the 4th point satisfies
        // the above equation
        float delta = 0.001f;
        if (-1 * delta < equation && equation < delta)
        //if (equation == 0)
            return true; // coplanar
        else
            return false; // not coplanar
    } 
    bool convert_to_3d(k4a_calibration_t &calibration,
                       Mat &depthMat,
                       vector<Point2f> &points_2d,
                       vector<Point3f> &points_3d)
    {
        points_3d.clear();
        for (int i = 0; i < points_2d.size(); i++)
        {
            k4a_float2_t point_2d_depth;
            k4a_float3_t point_3d_depth;

            point_2d_depth.xy.x = static_cast<float>(points_2d[i].x);
            point_2d_depth.xy.y = static_cast<float>(points_2d[i].y);

            int idx_col = (int)points_2d[i].x;
            int idx_row = (int)points_2d[i].y;
            float point_2d_depth_value = depthMat.at<uint16_t>(idx_row, idx_col);

            int valid;
            k4a_calibration_2d_to_3d(&calibration,
                                     &point_2d_depth,
                                     point_2d_depth_value,
                                     K4A_CALIBRATION_TYPE_DEPTH,
                                     K4A_CALIBRATION_TYPE_DEPTH,
                                     &point_3d_depth,
                                     &valid);
            if (!valid)
                return false;
            points_3d.push_back(Point3f(point_3d_depth.xyz.x, point_3d_depth.xyz.y, point_3d_depth.xyz.z));
        }
        return true;
    }
    bool get_plane_block(Mat &depthMat,
                         int row,
                         int col,
                         vector<Point2f> &corners,
                         vector<Point2f> &inners,
                         int block_size = 10)
    {
        for (int i = 0; i < block_size; i++)
            for (int j = 0; j < block_size; j++)
                if (depthMat.at<uint16_t>(row + i, col + j) == 0)
                    return false;

        corners.clear();
        inners.clear();

        corners.push_back(Point2f((float)col + block_size - 1, (float)row + block_size - 1));
        corners.push_back(Point2f((float)col + block_size - 1, (float)row + 0));
        corners.push_back(Point2f((float)col + 0, (float)row + 0));
        corners.push_back(Point2f((float)col + 0, (float)row + block_size - 1));

        if (block_size > 3)
        {
            int inner_min = block_size / 4;
            int inner_center = block_size / 2;
            int inner_max = 3 * block_size / 4;

            inners.push_back(Point2f((float)col + inner_max, (float)row + inner_max));
            inners.push_back(Point2f((float)col + inner_max, (float)row + inner_min));
            inners.push_back(Point2f((float)col + inner_center, (float)row + inner_center));
            inners.push_back(Point2f((float)col + inner_min, (float)row + inner_min));
            inners.push_back(Point2f((float)col + inner_min, (float)row + inner_max));
        }
        else if (block_size == 3)
        {
            inners.push_back(Point2f((float)col + 1, (float)row));
            inners.push_back(Point2f((float)col, (float)row + 1));
            inners.push_back(Point2f((float)col + 1, (float)row + 1));
            inners.push_back(Point2f((float)col + 2, (float)row + 1));
            inners.push_back(Point2f((float)col + 1, (float)row + 2));
        }
        else
            return false;
        

        return true;
    }
    void find_plane(k4a_calibration_t &calibration,
                    Mat &depthMat,
                    vector<vector<Point2f>> &planes,
                    vector<vector<Point3f>> &planes_3d,
                    int block_size = 10)
    {
        int rows = depthMat.rows;
        int cols = depthMat.cols;
        vector<Point2f> corners, inners;
        vector<Point3f> corners_3d, inners_3d;
        for (int r = 0; r < rows - block_size; r++)
        {
            for (int c = 0; c < cols - block_size; c++)
            {
                if (!get_plane_block(depthMat, r, c, corners, inners, block_size))
                    continue;
                if (!convert_to_3d(calibration, depthMat, corners, corners_3d))
                    continue;
                if (!convert_to_3d(calibration, depthMat, inners, inners_3d))
                    continue;

                bool is_coplanar = plane_equation(corners_3d[0], corners_3d[1], corners_3d[2], corners_3d[3]);
                if (is_coplanar)
                {
                    /*for (int i = 0; i < inners_3d.size(); i++)
                    {
                        bool plane = plane_equation(corners_3d[0], corners_3d[1], corners_3d[2], inners_3d[i]);
                        if (!plane)
                        {
                            is_coplanar = false;
                            break;
                        }
                    }*/
                    if (is_coplanar)
                    {
                        planes.push_back(corners);
                        planes_3d.push_back(corners_3d);
                    }
                }
            }
        }
    }

    void homography(vector<Point2f> &corners, Mat &input_image, Mat &warped_image, int width = 640, int height = 480, int offset = 0) {
        vector<Point> new_corners;
        new_corners.push_back(Point(width + offset, height + offset));
        new_corners.push_back(Point(width + offset, 0 + offset));
        new_corners.push_back(Point(0 + offset, 0 + offset));
        new_corners.push_back(Point(0 + offset, height + offset));
        Mat H = findHomography(corners, new_corners);
        warpPerspective(input_image, warped_image, H, Size(width + 2 * offset, height + 2 * offset));
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

    float get_depth_of_color_pixel(k4a_calibration_t &calibration,
                                   k4a_image_t &depth_image,
                                   Mat &depthMat,
                                   k4a_float2_t point_2d)
    {
        k4a_float2_t point_2d_depth;
        int valid;
        k4a_calibration_color_2d_to_depth_2d(&calibration, &point_2d, depth_image, &point_2d_depth, &valid);
        int idx_col = (int)round(point_2d_depth.xy.x);
        int idx_row = (int)round(point_2d_depth.xy.y);
        if (!valid)
            return 0;
        return depthMat.at<uint16_t>(idx_row, idx_col);
    }

    void raycast(k4a_calibration_t &calibration,
                 k4a_image_t &depth_image,
                 Mat &depthMat,
                 vector<Point2f> &old_corners,
                 vector<Point2f> &new_corners)
    {
        new_corners.clear();
        for (int i = 0; i < old_corners.size(); i++)
        {
            k4a_float2_t point_2d, point_2d_projected;
            k4a_float3_t point_3d, point_3d_depth;
            int valid;

            // Detected 2d Charuco corner point
            point_2d.xy.x = static_cast<float>(old_corners[i].x);
            point_2d.xy.y = static_cast<float>(old_corners[i].y);

            float point_2d_depth_value = get_depth_of_color_pixel(calibration, depth_image, depthMat, point_2d);

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

            new_corners.push_back(Point2f(point_2d_projected.xy.x, point_2d_projected.xy.y));
        }
    }

    void color_corners_of_depth_planes(k4a_calibration_t &calibration,
                                       vector<Point3f> &old_corners_3d,
                                       vector<Point2f> &new_corners)
    {
        new_corners.clear();
        for (int i = 0; i < old_corners_3d.size(); i++)
        {
            k4a_float2_t point_2d_projected;
            k4a_float3_t point_3d;
            int valid;

            point_3d.xyz.x = static_cast<float>(old_corners_3d[i].x);
            point_3d.xyz.y = static_cast<float>(old_corners_3d[i].y);
            point_3d.xyz.z = static_cast<float>(old_corners_3d[i].z);

            // Project new 3d point back to 2d
            k4a_calibration_3d_to_2d(&calibration,
                                     &point_3d,
                                     K4A_CALIBRATION_TYPE_DEPTH,
                                     K4A_CALIBRATION_TYPE_COLOR,
                                     &point_2d_projected,
                                     &valid);

            new_corners.push_back(Point2f(point_2d_projected.xy.x, point_2d_projected.xy.y));
        }
    }
    void project_planes_direct(k4a_calibration_t &calibration,
                               vector<Point3f> &old_corners_3d,
                               vector<Point2f> &new_corners)
    {
        new_corners.clear();
        for (int i = 0; i < old_corners_3d.size(); i++)
        {
            k4a_float2_t point_2d_projected;
            k4a_float3_t point_3d, point_3d_projected;
            int valid;

            point_3d.xyz.x = static_cast<float>(old_corners_3d[i].x);
            point_3d.xyz.y = static_cast<float>(old_corners_3d[i].y);
            point_3d.xyz.z = static_cast<float>(old_corners_3d[i].z);

            // Multiply 3d point with camera2camera matrix
            linmath::vec4 point_4d{ point_3d.xyz.x, point_3d.xyz.y, point_3d.xyz.z, 1.0f },
                projected_point_4d;
            linmath::mat4x4_mul_vec4(projected_point_4d, m_c2c_mm, point_4d);

            // Remove last dimension
            point_3d_projected.xyz.x = static_cast<float>(projected_point_4d[0]);
            point_3d_projected.xyz.y = static_cast<float>(projected_point_4d[1]);
            point_3d_projected.xyz.z = static_cast<float>(projected_point_4d[2]);

            // Project new 3d point back to 2d
            k4a_calibration_3d_to_2d(&calibration,
                                     &point_3d_projected,
                                     K4A_CALIBRATION_TYPE_DEPTH,
                                     K4A_CALIBRATION_TYPE_COLOR,
                                     &point_2d_projected,
                                     &valid);

            new_corners.push_back(Point2f(point_2d_projected.xy.x, point_2d_projected.xy.y));
        }
    }
    bool project_planes(k4a_calibration_t &calibration_1,
                        k4a_image_t &depth_image,
                        Mat &depthMat,
                        k4a_calibration_t &calibration_2,
                        vector<Point3f> &old_corners_3d,
                        vector<Point2f> &new_corners)
    {
        new_corners.clear();
        for (int i = 0; i < old_corners_3d.size(); i++)
        {
            k4a_float2_t point_2d_projected;
            k4a_float3_t point_3d, point_3d_projected, point_3d_color, point_3d_reprojected;
            int valid;

            point_3d.xyz.x = static_cast<float>(old_corners_3d[i].x);
            point_3d.xyz.y = static_cast<float>(old_corners_3d[i].y);
            point_3d.xyz.z = static_cast<float>(old_corners_3d[i].z);

            k4a_calibration_3d_to_3d(&calibration_2,
                                     &point_3d,
                                     K4A_CALIBRATION_TYPE_DEPTH,
                                     K4A_CALIBRATION_TYPE_COLOR,
                                     &point_3d_color);

            // Multiply 3d point with camera2camera matrix
            linmath::vec4 point_4d{ point_3d_color.xyz.x, point_3d_color.xyz.y, point_3d_color.xyz.z, 1.0f },
                projected_point_4d;
            linmath::mat4x4_mul_vec4(projected_point_4d, m_c2c_mm, point_4d);

            // Remove last dimension
            point_3d_projected.xyz.x = static_cast<float>(projected_point_4d[0]);
            point_3d_projected.xyz.y = static_cast<float>(projected_point_4d[1]);
            point_3d_projected.xyz.z = static_cast<float>(projected_point_4d[2]);

            // Project new 3d point back to 2d
            k4a_calibration_3d_to_2d(&calibration_1,
                                     &point_3d_projected,
                                     K4A_CALIBRATION_TYPE_COLOR,
                                     K4A_CALIBRATION_TYPE_COLOR,
                                     &point_2d_projected,
                                     &valid);

            // occlusion check
            float point_2d_depth_value =
                get_depth_of_color_pixel(calibration_1, depth_image, depthMat, point_2d_projected);

            // Project 2d point to 3d point
            k4a_calibration_2d_to_3d(&calibration_1,
                                     &point_2d_projected,
                                     point_2d_depth_value,
                                     K4A_CALIBRATION_TYPE_COLOR,
                                     K4A_CALIBRATION_TYPE_COLOR,
                                     &point_3d_reprojected,
                                     &valid);

            Point3f projected_3d(point_3d_projected.xyz.x, point_3d_projected.xyz.y, point_3d_projected.xyz.z);
            Point3f reprojected_3d(point_3d_reprojected.xyz.x, point_3d_reprojected.xyz.y, point_3d_reprojected.xyz.z);
            double distance = norm(projected_3d-reprojected_3d);
            if (distance > 50)
                return false;
            new_corners.push_back(Point2f(point_2d_projected.xy.x, point_2d_projected.xy.y));
        }
        return true;
    }

    void project_2d_to_2d(k4a_calibration_t &calibration_1,
                          k4a_calibration_t &calibration_2,
                          k4a_image_t &depth_image,
                          Mat &depthMat,
                          vector<Point2f> &old_corners,
                          vector<Point2f> &new_corners)
    {
        new_corners.clear();
        for (int i = 0; i < old_corners.size(); i++)
        {
            k4a_float2_t point_2d, point_2d_projected;
            k4a_float3_t point_3d, point_3d_projected;
            int valid;

            // Detected 2d Charuco corner point
            point_2d.xy.x = static_cast<float>(old_corners[i].x);
            point_2d.xy.y = static_cast<float>(old_corners[i].y);
            
            float point_2d_depth_value = get_depth_of_color_pixel(calibration_2, depth_image, depthMat, point_2d);

            // Project 2d point to 3d point
            k4a_calibration_2d_to_3d(&calibration_2,
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
            k4a_calibration_3d_to_2d(&calibration_1,
                                     &point_3d_projected,
                                     K4A_CALIBRATION_TYPE_COLOR,
                                     K4A_CALIBRATION_TYPE_COLOR,
                                     &point_2d_projected,
                                     &valid);

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
        Mat empty;
        return error_by_frame_per_point(
            vector<Point2f>(), corners_3d_1, ids_1, vector<Point2f>(), corners_3d_2, ids_2, true, empty);
    }
    double pixel_error_by_frame_per_point(vector<Point2f> corners_1,
                                          vector<int> ids_1,
                                          vector<Point2f> corners_2,
                                          vector<int> ids_2,
                                          Mat &image)
    {
        return error_by_frame_per_point(corners_1, vector<Point3f>(), ids_1, corners_2, vector<Point3f>(), ids_2, false, image);
    }

    double error_by_frame_per_point(vector<Point2f> corners_1,
                                    vector<Point3f> corners_3d_1,
                                    vector<int> ids_1,
                                    vector<Point2f> corners_2,
                                    vector<Point3f> corners_3d_2,
                                    vector<int> ids_2,
                                    bool error_in_3d,
                                    Mat &image)
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
                    arrowedLine(image, corners_1[idx_1], corners_2[idx_2], Scalar(255, 0, 255), 2, LINE_AA, 0, 0.3);
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

    vector<Point2f> m_temp_corners_1, m_temp_corners_2;
    vector<Point3f> m_temp_corners_3d_1, m_temp_corners_3d_2;
    vector<vector<Point2f>> m_planes_2;
    vector<vector<Point3f>> m_planes_3d_2;
    vector<Point2f> m_hom_corners_1, m_hom_corners_2;

    linmath::mat4x4 m_c2c, m_c2c_mm;
    Charuco m_charuco_1, m_charuco_2;

    Video video_1, video_2;

    Mat m_warpMat_1, m_warpMat_2;
};
} // namespace k4aviewer

#endif
