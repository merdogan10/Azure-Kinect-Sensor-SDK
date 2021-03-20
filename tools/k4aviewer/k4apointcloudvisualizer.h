// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef K4APOINTCLOUDVISUALIZER_H
#define K4APOINTCLOUDVISUALIZER_H

// System headers
//
#include <vector>

// Library headers
//
#include <k4a/k4a.hpp>

// Project headers
//
#include "gpudepthtopointcloudconverter.h"
#include "k4apointcloudrenderer.h"
#include "k4apointcloudviewcontrol.h"
#include "k4aviewerutil.h"
#include "k4aviewerimage.h"

namespace k4aviewer
{

enum class PointCloudVisualizationResult
{
    Success,
    OpenGlError,
    MissingDepthImage,
    MissingColorImage,
    DepthToXyzTransformationFailed,
    DepthToColorTransformationFailed
};

class K4APointCloudVisualizer
{
public:
    enum class ColorizationStrategy
    {
        Simple,
        Shaded,
        Color
    };

    GLenum InitializeTexture(std::shared_ptr<K4AViewerImage> *texture) const;
    PointCloudVisualizationResult UpdateTexture(std::shared_ptr<K4AViewerImage> *texture,
                                                const k4a::capture &capture,
                                                const k4a::capture &capture2);

    void ProcessMouseMovement(const linmath::vec2 displayDimensions,
                              const linmath::vec2 mousePos,
                              const linmath::vec2 mouseDelta,
                              MouseMovementType movementType);
    void ProcessMouseScroll(float yoffset);

    void ResetPosition();

    PointCloudVisualizationResult SetColorizationStrategy(ColorizationStrategy strategy);
    void SetPointSize(int size);

    K4APointCloudVisualizer(bool enableColorPointCloud,
                            const k4a::calibration &calibrationData,
                            const k4a::calibration &calibrationData2,
                            linmath::mat4x4 se3_depth,
                            linmath::mat4x4 se3_color);
    ~K4APointCloudVisualizer() = default;

    K4APointCloudVisualizer(const K4APointCloudVisualizer &) = delete;
    K4APointCloudVisualizer &operator=(const K4APointCloudVisualizer &) = delete;
    K4APointCloudVisualizer(const K4APointCloudVisualizer &&) = delete;
    K4APointCloudVisualizer &operator=(const K4APointCloudVisualizer &&) = delete;

private:
    PointCloudVisualizationResult UpdatePointClouds(const k4a::capture &capture,
                                                    PointCloudRenderer &pointCloudRenderer,
                                                    k4a::transformation &transformation,
                                                    k4a::image &transformedDepthImage,
                                                    OpenGL::Texture &xyzTexture,
                                                    k4a::capture &lastCapture,
                                                    k4a::image &pointCloudColorization,
                                                    GpuDepthToPointCloudConverter &pointCloudConverter,
                                                    int &color);

    PointCloudVisualizationResult
    K4APointCloudVisualizer::SetColorizationStrategyIn(PointCloudRenderer &pointCloudRenderer,
                                                       k4a::transformation &transformation,
                                                       k4a::image &transformedDepthImage,
                                                       OpenGL::Texture &xyzTexture,
                                                       k4a::capture &lastCapture,
                                                       k4a::image &pointCloudColorization,
                                                       GpuDepthToPointCloudConverter &pointCloudConverter,
                                                       const k4a::calibration &calibrationData,
                                                       k4a::image &colorXyTable,
                                                       k4a::image &depthXyTable,
                                                       int &color);

    std::pair<DepthPixel, DepthPixel> m_expectedValueRange;
    ImageDimensions m_dimensions;

    PointCloudRenderer m_pointCloudRenderer;
    PointCloudRenderer m_pointCloudRenderer2;
    ViewControl m_viewControl;

    bool m_enableColorPointCloud = false;
    ColorizationStrategy m_colorizationStrategy;

    linmath::mat4x4 m_projection{};
    linmath::mat4x4 m_view{};
    linmath::mat4x4 m_se3_depth;
    linmath::mat4x4 m_se3_color;

    OpenGL::Framebuffer m_frameBuffer = OpenGL::Framebuffer(true);
    OpenGL::Renderbuffer m_depthBuffer = OpenGL::Renderbuffer(true);

    k4a::calibration m_calibrationData, m_calibrationData2;
    k4a::transformation m_transformation, m_transformation2;

    k4a::capture m_lastCapture, m_lastCapture2;

    // Buffer that holds the depth image transformed to the color coordinate space.
    // Used in color mode only.
    //
    k4a::image m_transformedDepthImage;
    k4a::image m_transformedDepthImage2;

    // In color mode, this is just a shallow copy of the latest color image.
    // In depth mode, this is a buffer that holds the colorization of the depth image.
    //
    k4a::image m_pointCloudColorization;
    k4a::image m_pointCloudColorization2;

    // Holds the XYZ point cloud as a texture.
    // Format is XYZA, where A (the alpha channel) is unused.
    //
    OpenGL::Texture m_xyzTexture, m_xyzTexture2;

    int m_color = 1, m_color2 = 2; // 0: Blue, 1: Green, 2: Red

    GpuDepthToPointCloudConverter m_pointCloudConverter, m_pointCloudConverter2;

    k4a::image m_colorXyTable;
    k4a::image m_colorXyTable2;
    k4a::image m_depthXyTable;
    k4a::image m_depthXyTable2;
};
} // namespace k4aviewer

#endif
