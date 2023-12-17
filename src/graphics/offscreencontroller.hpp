#pragma once

#include <EGL/egl.h>
#include <string>


class OffscreeenController
{
public:
    OffscreeenController();
    ~OffscreeenController();

    int width, height;

private:
    EGLDisplay eglDpy;
    EGLContext eglCtx;
    EGLSurface eglSurf;

    void assertEGLError(const std::string& msg);
};
