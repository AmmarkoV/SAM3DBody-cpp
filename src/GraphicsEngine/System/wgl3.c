/* Win32/WGL implementation of the existing glx3.h window interface.
 * Adapted from beemsoft's Windows port (MIT), PR #13:
 * https://github.com/AmmarkoV/SAM3DBody-cpp/pull/13
 * Extended for OpenGL 3.3, the current title/callback API, and checked teardown.
 */
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <windowsx.h>
#include <GL/gl.h>
#include <stdio.h>
#include <stdint.h>
#include <wchar.h>
#include "glx3.h"

#define WGL_CONTEXT_MAJOR_VERSION_ARB 0x2091
#define WGL_CONTEXT_MINOR_VERSION_ARB 0x2092
#define WGL_CONTEXT_PROFILE_MASK_ARB  0x9126
#define WGL_CONTEXT_CORE_PROFILE_BIT_ARB 0x00000001

typedef HGLRC (WINAPI *CreateContextAttribsProc)(HDC, HGLRC, const int *);
typedef BOOL (WINAPI *SwapIntervalProc)(int);

extern int handleUserInput(int key, int state, int x, int y);
extern int windowSizeUpdated(unsigned int width, unsigned int height);

static const wchar_t class_name[] = L"SAM3DBodyWGLWindow";
static const wchar_t default_title[] = L"SAM3DBody-cpp OpenGL3.x+ Visualization";
static wchar_t window_title[256] = L"SAM3DBody-cpp OpenGL3.x+ Visualization";
static HWND window_handle = NULL;
static HDC device_context = NULL;
static HGLRC render_context = NULL;
static int close_requested = 0;
static int swap_interval = 1;

/* Some Windows drivers return a non-NULL sentinel for an absent extension. */
static PROC get_wgl_proc(const char *name)
{
    PROC proc = wglGetProcAddress(name);
    intptr_t value = (intptr_t)proc;
    return (value == 0 || value == 1 || value == 2 || value == 3 || value == -1)
           ? NULL : proc;
}

static LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM key, LPARAM data)
{
    switch (message) {
    case WM_CLOSE:
        /* Keep the surface alive until the renderer saves outputs and tears down. */
        glx3_request_close();
        return 0;
    case WM_DESTROY:
        glx3_request_close();
        return 0;
    case WM_KEYDOWN:
    case WM_KEYUP:
        if (key == VK_ESCAPE && message == WM_KEYDOWN) glx3_request_close();
        else handleUserInput((int)key, message == WM_KEYDOWN, 0, 0);
        return 0;
    case WM_LBUTTONDOWN:
    case WM_LBUTTONUP:
    case WM_MBUTTONDOWN:
    case WM_MBUTTONUP:
    case WM_RBUTTONDOWN:
    case WM_RBUTTONUP:
    {
        int button = (message == WM_LBUTTONDOWN || message == WM_LBUTTONUP) ? 1 :
                     (message == WM_MBUTTONDOWN || message == WM_MBUTTONUP) ? 2 : 3;
        int pressed = message == WM_LBUTTONDOWN || message == WM_MBUTTONDOWN ||
                      message == WM_RBUTTONDOWN;
        POINT position = { GET_X_LPARAM(data), GET_Y_LPARAM(data) };
        ClientToScreen(window, &position);
        handleUserInput(button, pressed, position.x, position.y);
        return 0;
    }
    case WM_SIZE:
        if (key != SIZE_MINIMIZED) windowSizeUpdated(LOWORD(data), HIWORD(data));
        return 0;
    case WM_ERASEBKGND:
        return 1;
    }
    return DefWindowProcW(window, message, key, data);
}

void glx3_set_window_title(const char *title)
{
    const int capacity = (int)(sizeof(window_title) / sizeof(window_title[0]));
    if (title && *title) {
        if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                               title, -1, window_title, capacity)) return;
        /* main(char **) receives ANSI argv on Windows unless the executable
         * explicitly opts into UTF-8. Preserve native CLI titles as well. */
        if (MultiByteToWideChar(CP_ACP, 0, title, -1, window_title, capacity)) return;
    }
    wcscpy_s(window_title, capacity, default_title);
}

int disableVSync(void)
{
    swap_interval = 0;
    if (render_context) {
        SwapIntervalProc swap = (SwapIntervalProc)get_wgl_proc("wglSwapIntervalEXT");
        return swap ? (int)swap(0) : 0;
    }
    return 1; /* Apply the request after context creation. */
}

int stop_glx3_stuff(void)
{
    if (render_context) {
        wglMakeCurrent(NULL, NULL);
        wglDeleteContext(render_context);
        render_context = NULL;
    }
    if (device_context) {
        ReleaseDC(window_handle, device_context);
        device_context = NULL;
    }
    if (window_handle) {
        DestroyWindow(window_handle);
        window_handle = NULL;
    }
    UnregisterClassW(class_name, GetModuleHandleW(NULL));
    close_requested = 1;
    return 1;
}

int start_glx3_stuff(int width, int height, int viewWindow, int argc, const char **argv)
{
    HINSTANCE instance = GetModuleHandleW(NULL);
    WNDCLASSW window_class = {0};
    PIXELFORMATDESCRIPTOR pixel_format = {0};
    RECT rectangle = {0, 0, width, height};
    DWORD style = viewWindow ? WS_OVERLAPPEDWINDOW : WS_POPUP;
    CreateContextAttribsProc create_context;
    HGLRC modern_context;
    int format;
    const int context_attributes[] = {
        WGL_CONTEXT_MAJOR_VERSION_ARB, 3,
        WGL_CONTEXT_MINOR_VERSION_ARB, 3,
        WGL_CONTEXT_PROFILE_MASK_ARB, WGL_CONTEXT_CORE_PROFILE_BIT_ARB, 0
    };
    (void)argc;
    (void)argv;

    if (window_handle || render_context) stop_glx3_stuff();
    close_requested = 0;
    if (width <= 0 || height <= 0) {
        fprintf(stderr, "WGL: invalid surface size %dx%d\n", width, height);
        close_requested = 1;
        return 0;
    }

    window_class.style = CS_OWNDC;
    window_class.lpfnWndProc = window_proc;
    window_class.hInstance = instance;
    window_class.lpszClassName = class_name;
    window_class.hCursor = LoadCursorW(NULL, MAKEINTRESOURCEW(32512)); /* IDC_ARROW */
    if (!RegisterClassW(&window_class)) goto fail;
    if (!AdjustWindowRect(&rectangle, style, FALSE)) goto fail;
    window_handle = CreateWindowExW(0, class_name, window_title, style,
        CW_USEDEFAULT, CW_USEDEFAULT, rectangle.right - rectangle.left,
        rectangle.bottom - rectangle.top, NULL, NULL, instance, NULL);
    if (!window_handle) goto fail;
    device_context = GetDC(window_handle);
    if (!device_context) goto fail;

    pixel_format.nSize = sizeof(pixel_format);
    pixel_format.nVersion = 1;
    pixel_format.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
    pixel_format.iPixelType = PFD_TYPE_RGBA;
    pixel_format.cColorBits = 32;
    pixel_format.cDepthBits = 24;
    pixel_format.cStencilBits = 8;
    pixel_format.iLayerType = PFD_MAIN_PLANE;
    format = ChoosePixelFormat(device_context, &pixel_format);
    if (!format || !SetPixelFormat(device_context, format, &pixel_format)) goto fail;

    /* A temporary legacy context is needed to resolve the modern WGL entrypoint. */
    render_context = wglCreateContext(device_context);
    if (!render_context || !wglMakeCurrent(device_context, render_context)) goto fail;
    create_context = (CreateContextAttribsProc)get_wgl_proc("wglCreateContextAttribsARB");
    if (!create_context) {
        fprintf(stderr, "WGL: OpenGL 3.3 is required; install the GPU vendor's display driver.\n");
        goto fail;
    }
    modern_context = create_context(device_context, NULL, context_attributes);
    if (!modern_context) goto fail;
    wglMakeCurrent(NULL, NULL);
    wglDeleteContext(render_context);
    render_context = modern_context;
    if (!wglMakeCurrent(device_context, render_context)) goto fail;

    {
        SwapIntervalProc swap = (SwapIntervalProc)get_wgl_proc("wglSwapIntervalEXT");
        if (swap) swap(viewWindow ? swap_interval : 0);
    }
    /* Headless mode still uses an unshown Win32 window, with no mapped surface. */
    if (viewWindow) ShowWindow(window_handle, SW_SHOW);
    fprintf(stderr, "WGL OpenGL %s context ready (%s)\n", glGetString(GL_VERSION),
            viewWindow ? "windowed" : "hidden");
    return 1;

fail:
    fprintf(stderr, "WGL context creation failed (Win32 error %lu).\n", GetLastError());
    stop_glx3_stuff();
    return 0;
}

int glx3_endRedraw(void)
{
    return device_context ? (int)SwapBuffers(device_context) : 0;
}

int glx3_should_close(void) { return close_requested; }
void glx3_request_close(void) { close_requested = 1; }

int glx3_checkEvents(void)
{
    MSG message;
    if (!window_handle || close_requested) return 0;
    while (PeekMessageW(&message, NULL, 0, 0, PM_REMOVE)) {
        if (message.message == WM_QUIT) glx3_request_close();
        else {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }
        if (close_requested) return 0;
    }
    return 1;
}
