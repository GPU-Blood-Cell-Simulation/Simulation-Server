#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <vector>
#include <glad/glad.h>


class StremmingController
{
public:
    StremmingController();
    ~StremmingController();

    void SendFrame();
private:
    GstElement *pipeline, *appsrc, *conv, *videosink;

    std::vector<GLubyte> pixels;
    GstClockTime timestamp;
};
