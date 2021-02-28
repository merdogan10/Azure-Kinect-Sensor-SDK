// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef CHARUCO_H
#define CHARUCO_H

// System headers
//

// Library headers
//
#include <opencv2/aruco/charuco.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include "linmath.h"

// Project headers
//
#include "utils.h"
using namespace cv;
using namespace std;
namespace k4aviewer
{
class Charuco
{
public:

    Charuco(string input_video)
    {
        intrinsics_extrinsics(input_video.c_str(), m_cameraMatrix, m_distCoeffs, m_extrinsics);
        init_board(m_board_type, m_dictionary, m_board, m_params);
    }
    Charuco(){};
    ~Charuco() = default;

    /**
     * \brief initializes charuco board
     */
    void init_board() {
        init_board(m_board_type, m_dictionary, m_board, m_params);
    }

    /**
     * \brief Charuco Pose detector when only have image path
     */
    void detect_charuco_pose(string input_video_path) {
        VideoCapture inputVideo;
        inputVideo.open(input_video_path);

        inputVideo.grab();
        Mat image;
        inputVideo.retrieve(image);
        detect_charuco_pose(image);
    }

    /**
    * \brief Charuco Pose detector when only have image
    */
    void detect_charuco_pose(Mat image) {
        detect_charuco_pose(
            image, m_board, m_params, m_cameraMatrix, m_distCoeffs, m_detected_corners, m_detected_ids, m_rvec, m_tvec);
    }
    
    Vec3d m_rvec, m_tvec;
    vector<Point2f> m_detected_corners;
    vector<int> m_detected_ids;
    Mat m_cameraMatrix, m_distCoeffs;
    linmath::mat4x4 m_extrinsics;

private:
    /**
     * \brief initializes charuco board
     *
     * only boards with 5x7, 14x9, 4x3 are supported. 4x3 board is the default one.
     */
    void init_board(int board_type,
                    Ptr<aruco::Dictionary> &dictionary,
                    Ptr<aruco::CharucoBoard> &board,
                    Ptr<aruco::DetectorParameters> &params)
    {
        switch (board_type)
        {
        case 5:
            dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
            board = aruco::CharucoBoard::create(5, 7, 0.04f, 0.02f, dictionary);
            break;
        case 14:
            dictionary = aruco::getPredefinedDictionary(aruco::DICT_5X5_250);
            board = aruco::CharucoBoard::create(14, 9, 0.02f, 0.01556f, dictionary);
            break;
        case 4:
            dictionary = aruco::getPredefinedDictionary(aruco::DICT_4X4_250);
            board = aruco::CharucoBoard::create(4, 3, 0.053f, 0.04f, dictionary);
            break;
        default:
            throw std::runtime_error("Invalid board!");
            break;
        }
        params = aruco::DetectorParameters::create();
    }

    /**
     * \brief standalone Charuco Pose Detector
     */
    void detect_charuco_pose(Mat image,
                             Ptr<aruco::CharucoBoard> &board,
                             Ptr<aruco::DetectorParameters> &params,
                             Mat &cameraMatrix,
                             Mat &distCoeffs,
                             vector<Point2f> &detected_corners,
                             vector<int> &detected_ids,
                             Vec3d &rvec,
                             Vec3d &tvec)
    {
        vector<int> markerIds;
        vector<vector<Point2f>> markerCorners;
        aruco::detectMarkers(image, board->dictionary, markerCorners, markerIds, params);
        // if at least one marker detected
        if (markerIds.size() > 0)
        {
            aruco::interpolateCornersCharuco(
                markerCorners, markerIds, image, board, detected_corners, detected_ids, cameraMatrix, distCoeffs);
            // if at least one corner detected
            if (detected_ids.size() > 0)
            {
                aruco::estimatePoseCharucoBoard(
                    detected_corners, detected_ids, board, cameraMatrix, distCoeffs, rvec, tvec);
            }
        }
    }

    Ptr<aruco::Dictionary> m_dictionary;
    Ptr<aruco::CharucoBoard> m_board;
    Ptr<aruco::DetectorParameters> m_params;
    int m_board_type = 4; // Change board_type according to your charuco board
};

} // namespace k4aviewer

#endif
