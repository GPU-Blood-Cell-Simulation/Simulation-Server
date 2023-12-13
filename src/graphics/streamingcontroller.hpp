#pragma once

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>

#include <vector>
#include <glad/glad.h>
#include <string>


class StremmingController
{
public:
    StremmingController(const std::string& host, int port);
    ~StremmingController();

    void StartStreaming();
    void StopStreaming();

    void SendFrame();
private:
    const std::string host;
    const int port;
    GstElement *pipeline, *appsrc, *x264enc, *rtph264pay, *udpsink, *convert,
        *queue;
    GstBus *bus;

    std::vector<GLubyte> pixels;
    GstClockTime timestamp;

    //void bus_message_handler(GstBus *bus, GstMessage *msg, gpointer data);
};
