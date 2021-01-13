// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Associated header
//
#include "k4asourceselectiondockcontrol.h"

// System headers
//
#include <sstream>

// Library headers
//
#include "k4aimgui_all.h"
#include <k4a/k4a.hpp>
#include <k4arecord/playback.hpp>

// Project headers
//
#include "filesystem17.h"
#include "k4aaudiomanager.h"
#include "k4aimguiextensions.h"
#include "k4aviewererrormanager.h"
#include "k4arecordingdockcontrol.h"
#include "k4aviewerutil.h"
#include "k4awindowmanager.h"
#include "charuco.h"
#include "linmath.h"
using namespace k4aviewer;

K4ASourceSelectionDockControl::K4ASourceSelectionDockControl()
{
    RefreshDevices();
}

K4ADockControlStatus K4ASourceSelectionDockControl::Show()
{
    ImGui::SetNextTreeNodeOpen(true, ImGuiCond_FirstUseEver);
    if (ImGui::TreeNode("Open Device"))
    {
        ImGuiExtensions::K4AComboBox("Device S/N",
                                     "(No available devices)",
                                     ImGuiComboFlags_None,
                                     m_connectedDevices,
                                     &m_selectedDevice);

        if (ImGui::Button("Refresh Devices"))
        {
            RefreshDevices();
        }

        ImGui::SameLine();

        const bool openAvailable = !m_connectedDevices.empty();
        {
            ImGuiExtensions::ButtonColorChanger colorChanger(ImGuiExtensions::ButtonColor::Green, openAvailable);
            if (ImGuiExtensions::K4AButton("Open Device", openAvailable))
            {
                OpenDevice();
            }
        }

        ImGui::TreePop();
    }

    ImGui::NewLine();
    ImGui::Separator();
    ImGui::NewLine();

    if (ImGui::TreeNode("Open Recording"))
    {
        if (m_filePicker.Show())
        {
            OpenRecording(m_filePicker.GetPath());
        }

        ImGui::TreePop();
    }

    return K4ADockControlStatus::Ok;
}

void K4ASourceSelectionDockControl::RefreshDevices()
{
    m_selectedDevice = -1;

    const uint32_t installedDevices = k4a_device_get_installed_count();

    m_connectedDevices.clear();

    for (uint32_t i = 0; i < installedDevices; i++)
    {
        try
        {
            k4a::device device = k4a::device::open(i);
            m_connectedDevices.emplace_back(std::make_pair(i, device.get_serialnum()));
        }
        catch (const k4a::error &)
        {
            // We can't have 2 handles to the same device, and we need to open a device handle to check
            // its serial number, so we expect devices we already have open to fail here.  Ignore those.
            //
            continue;
        }
    }

    if (!m_connectedDevices.empty())
    {
        m_selectedDevice = m_connectedDevices[0].first;
    }

    const int audioRefreshStatus = K4AAudioManager::Instance().RefreshDevices();
    if (audioRefreshStatus != SoundIoErrorNone)
    {
        std::stringstream errorBuilder;
        errorBuilder << "Failed to refresh audio devices: " << soundio_strerror(audioRefreshStatus) << "!" << std::endl
                     << "Attempting to open microphones may fail!";

        K4AViewerErrorManager::Instance().SetErrorStatus(errorBuilder.str());
    }
}

void K4ASourceSelectionDockControl::OpenDevice()
{
    try
    {
        if (m_selectedDevice < 0)
        {
            K4AViewerErrorManager::Instance().SetErrorStatus("No device selected!");
            return;
        }

        k4a::device device = k4a::device::open(static_cast<uint32_t>(m_selectedDevice));
        K4AWindowManager::Instance().PushLeftDockControl(std14::make_unique<K4ADeviceDockControl>(std::move(device)));
    }
    catch (const k4a::error &e)
    {
        K4AViewerErrorManager::Instance().SetErrorStatus(e.what());
    }
}

void K4ASourceSelectionDockControl::OpenRecording(const std17::filesystem::path &path)
{
    try
    {
        path;
        string calibration_file3 =
            "C:\\Users\\Mustafa\\Desktop\\thesis\\captures\\vicon_calibration\\cn03\\world2camera.txt";
        string calibration_file6 =
            "C:\\Users\\Mustafa\\Desktop\\thesis\\captures\\vicon_calibration\\cn06\\world2camera.txt";
        
        string input_path = "C:\\Users\\Mustafa\\Desktop\\thesis\\captures\\charuco_rotate\\cn03\\k4a_record.mkv";
        k4a::playback recording = k4a::playback::open(input_path.c_str());

        string input_path2 = "C:\\Users\\Mustafa\\Desktop\\thesis\\captures\\charuco_rotate\\cn06\\k4a_record.mkv";
        k4a::playback recording2 = k4a::playback::open(input_path2.c_str());

        linmath::mat4x4 se3, se3_2;
        linmath::mat4x4 c2c, c2c_id;
        
        int case_no = 3;
        bool is_quat = true;
        bool rotate_x = true;

        Charuco board(input_path, calibration_file3, is_quat, rotate_x);
        Charuco board2(input_path2, calibration_file6, is_quat, rotate_x);
        
        // If camera2camera is used, use identity matrix for the other view
        linmath::mat4x4_identity(c2c_id);
        switch (case_no)
        {
        case 0:  // se3_2' * se3
            board.get_se3(se3);
            board2.get_se3_inverse(se3_2);
            linmath::mat4x4_mul(c2c, se3_2, se3);
            break;
        case 1:  // se3 * se3_2'
            board.get_se3(se3);
            board2.get_se3_inverse(se3_2);
            linmath::mat4x4_mul(c2c, se3, se3_2);
            break;
        case 2:  // se3_2 * se3'
            board.get_se3_inverse(se3);
            board2.get_se3(se3_2);
            linmath::mat4x4_mul(c2c, se3_2, se3);
            break;
        case 3:  // se3' * se3_2
            board.get_se3_inverse(se3);
            board2.get_se3(se3_2);
            linmath::mat4x4_mul(c2c, se3, se3_2);
            break;
        default:
            break;
        }

        // If world2camera is used, send both of them for the views
        //board.get_se3(se3);
        //board2.get_se3(se3_2);

        //board.get_se3_inverse(se3);
        //board.get_se3_inverse(se3_2);


        K4AWindowManager::Instance().PushLeftDockControl(std14::make_unique<K4ARecordingDockControl>(
            input_path, std::move(recording), std::move(recording2), c2c_id, c2c));
    }
    catch (const k4a::error &e)
    {
        K4AViewerErrorManager::Instance().SetErrorStatus(e.what());
    }
}
