// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef UNCERTAINTY_H
#define UNCERTAINTY_H

// System headers
//
#include <fstream>

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
class Uncertainty
{
public:
    Uncertainty(string input_video) : m_video(input_video, 0), m_charuco(input_video)
    {
        int number_of_frames = 440;
        vector<Point3f> centers_cv, centers_k4a_depth, centers_k4a_color;
        vector<int> valid_frames;
        for (int i = 0; i < number_of_frames; i++)
        {
            bool result = false;
            result = m_video.next_capture_image();
            if (!result)
                continue;

            m_charuco.detect_charuco_pose(m_video.m_colorMat);
            if (!m_charuco.m_valid)
                continue;

            m_charuco.calculate_corners();

            Point3f center_cv(0, 0, 0), center_dep(0, 0, 0), center_col(0, 0, 0);
            for (int j = 0; j < m_charuco.m_outer_corners.size(); j++)
            {
                k4a_float2_t point_2d;
                k4a_float3_t point_3d_dep, point_3d_col;
                int valid;

                // 2d Charuco corner point
                point_2d.xy.x = static_cast<float>(m_charuco.m_outer_corners[j].x);
                point_2d.xy.y = static_cast<float>(m_charuco.m_outer_corners[j].y);

                k4a_float2_t point_2d_depth;
                k4a_calibration_color_2d_to_depth_2d(&m_video.m_calibration, &point_2d, m_video.m_depth_image, &point_2d_depth, &valid);
                int idx_col = (int)round(point_2d_depth.xy.x);
                int idx_row = (int)round(point_2d_depth.xy.y);

                float point_2d_depth_value = m_video.m_depthMat.at<uint16_t>(idx_row, idx_col);

                // Project 2d point to 3d point
                k4a_calibration_2d_to_3d(&m_video.m_calibration,
                                         &point_2d,
                                         point_2d_depth_value,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         K4A_CALIBRATION_TYPE_DEPTH,
                                         &point_3d_dep,
                                         &valid);
                Point3f dep_p(point_3d_dep.xyz.x, point_3d_dep.xyz.y, point_3d_dep.xyz.z);
                center_dep += dep_p;

                // Project 2d point to 3d point
                k4a_calibration_2d_to_3d(&m_video.m_calibration,
                                         &point_2d,
                                         point_2d_depth_value,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         K4A_CALIBRATION_TYPE_COLOR,
                                         &point_3d_col,
                                         &valid);
                Point3f col_p(point_3d_col.xyz.x, point_3d_col.xyz.y, point_3d_col.xyz.z);
                center_col += col_p;

                center_cv += m_charuco.m_outer_corners_3d[j] * 1000;

            }
            center_cv.x /= m_charuco.m_outer_corners_3d.size();
            center_cv.y /= m_charuco.m_outer_corners_3d.size();
            center_cv.z /= m_charuco.m_outer_corners_3d.size();
            centers_cv.push_back(center_cv);

            center_col.x /= m_charuco.m_outer_corners_3d.size();
            center_col.y /= m_charuco.m_outer_corners_3d.size();
            center_col.z /= m_charuco.m_outer_corners_3d.size();
            centers_k4a_depth.push_back(center_col);

            center_dep.x /= m_charuco.m_outer_corners_3d.size();
            center_dep.y /= m_charuco.m_outer_corners_3d.size();
            center_dep.z /= m_charuco.m_outer_corners_3d.size();
            centers_k4a_color.push_back(center_dep);

            valid_frames.push_back(i);
            m_video.release_images();
        }
        string captures_folder = "C:\\Users\\Mustafa\\Desktop\\thesis\\captures\\";

        ofstream cv_out(captures_folder + "cv.txt");
        ofstream color_out(captures_folder + "color.txt");
        ofstream depth_out(captures_folder + "depth.txt");
        ofstream frames_out(captures_folder + "frames.txt");
        for (int i = 0; i < centers_cv.size(); i++)
        {
            cv_out << centers_cv[i] << endl;
            color_out << centers_k4a_color[i] << endl;
            depth_out << centers_k4a_depth[i] << endl;
            frames_out << valid_frames[i] << endl;
        }
        m_video.close_playback();
    }

    Charuco m_charuco;

    Video m_video;
};
} // namespace k4aviewer

#endif
