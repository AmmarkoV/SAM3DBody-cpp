#include <windows.h>
#include <GL/gl.h>
#include <stdio.h>
#include "win32_gl.h"

// Note: We use the same function names as glx3.c to avoid changing the caller.

static HWND hwnd = NULL;
static HDC hdc = NULL;
static HGLRC hglrc = NULL;
static int close_requested = 0;

static LRESULT CALLBACK WindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam) {
    switch (uMsg) {
        case WM_CLOSE:
            close_requested = 1;
            return 0;
        case WM_DESTROY:
            PostQuitMessage(0);
            return 0;
        case WM_KEYDOWN:
            if (wParam == VK_ESCAPE) {
                close_requested = 1;
            }
            return 0;
    }
    return DefWindowProc(hwnd, uMsg, wParam, lParam);
}

int disableVSync() {
    typedef BOOL (WINAPI * PFNWGLSWAPINTERVALEXTPROC) (int interval);
    PFNWGLSWAPINTERVALEXTPROC wglSwapIntervalEXT = (PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
    if (wglSwapIntervalEXT) {
        return wglSwapIntervalEXT(0);
    }
    return 0;
}

int start_glx3_stuff(int WIDTH, int HEIGHT, int viewWindow, int argc, const char **argv) {
    HINSTANCE hInstance = GetModuleHandle(NULL);
    WNDCLASS wc = {0};
    wc.lpfnWndProc = WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = "FSB_Render_Window";
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);

    if (!RegisterClass(&wc)) {
        fprintf(stderr, "Failed to register window class\n");
        return 0;
    }

    DWORD dwStyle = WS_OVERLAPPEDWINDOW;
    if (!viewWindow) {
        dwStyle = WS_POPUP; // Hidden-ish or at least no decorations
    }

    RECT rect = {0, 0, WIDTH, HEIGHT};
    AdjustWindowRect(&rect, dwStyle, FALSE);

    hwnd = CreateWindowEx(
        0, "FSB_Render_Window", "SAM-3D-Body Renderer",
        dwStyle,
        CW_USEDEFAULT, CW_USEDEFAULT,
        rect.right - rect.left, rect.bottom - rect.top,
        NULL, NULL, hInstance, NULL
    );

    if (!hwnd) {
        fprintf(stderr, "Failed to create window\n");
        return 0;
    }

    if (viewWindow) {
        ShowWindow(hwnd, SW_SHOW);
    } else {
        // We don't show the window, but we still need a context
        ShowWindow(hwnd, SW_HIDE);
    }

    hdc = GetDC(hwnd);

    PIXELFORMATDESCRIPTOR pfd = {0};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.cStencilBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;

    int format = ChoosePixelFormat(hdc, &pfd);
    SetPixelFormat(hdc, format, &pfd);

    hglrc = wglCreateContext(hdc);
    wglMakeCurrent(hdc, hglrc);

    return 1;
}

int stop_glx3_stuff() {
    if (hglrc) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(hglrc);
        hglrc = NULL;
    }
    if (hdc) {
        ReleaseDC(hwnd, hdc);
        hdc = NULL;
    }
    if (hwnd) {
        DestroyWindow(hwnd);
        hwnd = NULL;
    }
    return 1;
}

int glx3_endRedraw() {
    if (hdc) {
        SwapBuffers(hdc);
        return 1;
    }
    return 0;
}

int glx3_checkEvents() {
    MSG msg;
    while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            close_requested = 1;
        }
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }
    return !close_requested;
}

int glx3_should_close() {
    return close_requested;
}

void glx3_request_close() {
    close_requested = 1;
}
