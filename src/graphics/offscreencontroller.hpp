#pragma once

#include <EGL/egl.h>
#include <string>

/// <summary>
/// Controls streaming actions
/// </summary>
class OffscreenController
{
public:
    OffscreenController();
    ~OffscreenController();

    int width, height;

private:
    EGLDisplay eglDpy;
    EGLContext eglCtx;
    EGLSurface eglSurf;

    void assertEGLError(const std::string& msg);
};
