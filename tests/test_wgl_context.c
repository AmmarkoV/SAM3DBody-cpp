/* Model-free Windows graphics smoke test: a GLSL 3.3 draw, pixel readback,
 * UTF-8 title, resize notification, graceful close, and context recreation.
 * Pass --windowed to exercise a visible surface; the default stays hidden. */
#include <GL/glew.h>
#include <windows.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include "../src/GraphicsEngine/System/glx3.h"

static unsigned int last_width, last_height;
int handleUserInput(int key, int state, int x, int y)
{ (void)key; (void)state; (void)x; (void)y; return 1; }
int windowSizeUpdated(unsigned int width, unsigned int height)
{ last_width = width; last_height = height; return 1; }

#define CHECK(condition, message) do { if (!(condition)) { \
    fprintf(stderr, "FAIL: %s\n", message); stop_glx3_stuff(); return 1; } } while (0)

static GLuint compile_shader(GLenum kind, const char *source)
{
    GLuint shader = glCreateShader(kind);
    GLint success = 0;
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) { glDeleteShader(shader); return 0; }
    return shader;
}

int main(int argc, const char **argv)
{
    int visible = argc > 1 && strcmp(argv[1], "--windowed") == 0;
    const char *vertex_source = "#version 330 core\n"
        "const vec2 p[3]=vec2[3](vec2(-1,-1),vec2(3,-1),vec2(-1,3));"
        "void main(){gl_Position=vec4(p[gl_VertexID],0,1);}";
    const char *fragment_source = "#version 330 core\n"
        "out vec4 color; void main(){color=vec4(1,0,0,1);}";
    GLuint vertex, fragment, program, vao;
    GLint linked = 0;
    unsigned char pixel[4] = {0};
    HWND window;
    RECT client_area;
    wchar_t title[256];

    CHECK(!start_glx3_stuff(0, 64, 0, argc, argv), "reject invalid size");
    glx3_set_window_title("WGL smoke \xE6\xB5\x8B\xE8\xAF\x95");
    disableVSync();
    CHECK(start_glx3_stuff(64, 64, visible, argc, argv), "create first context");
    glewExperimental = GL_TRUE;
    CHECK(glewInit() == GLEW_OK && GLEW_VERSION_3_3, "OpenGL 3.3 entrypoints");
    while (glGetError() != GL_NO_ERROR) {} /* GLEW probes legacy extensions. */
    window = WindowFromDC(wglGetCurrentDC());
    CHECK(window != NULL, "context owns a window");
    GetWindowTextW(window, title, 256);
    CHECK(wcscmp(title, L"WGL smoke \x6D4B\x8BD5") == 0, "UTF-8 title");
    CHECK((IsWindowVisible(window) != 0) == visible, "requested visibility");

    vertex = compile_shader(GL_VERTEX_SHADER, vertex_source);
    fragment = compile_shader(GL_FRAGMENT_SHADER, fragment_source);
    CHECK(vertex && fragment, "compile GLSL 3.3");
    program = glCreateProgram();
    glAttachShader(program, vertex);
    glAttachShader(program, fragment);
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    CHECK(linked, "link GLSL program");
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glUseProgram(program);
    glViewport(0, 0, 64, 64);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glReadBuffer(GL_BACK);
    glReadPixels(32, 32, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, pixel);
    CHECK(glGetError() == GL_NO_ERROR, "draw and read without GL error");
    CHECK(pixel[0] > 240 && pixel[1] < 10 && pixel[2] < 10, "red triangle readback");
    CHECK(glx3_endRedraw(), "swap buffers");
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(program);
    glDeleteShader(vertex);
    glDeleteShader(fragment);

    CHECK(SetWindowPos(window, NULL, 0, 0, 96, 80, SWP_NOMOVE | SWP_NOZORDER), "resize window");
    CHECK(GetClientRect(window, &client_area), "read resized client area");
    CHECK(glx3_checkEvents() && last_width == (unsigned int)client_area.right &&
          last_height == (unsigned int)client_area.bottom, "resize callback matches client area");
    SendMessageW(window, WM_CLOSE, 0, 0);
    CHECK(glx3_should_close() && !glx3_checkEvents(), "window close exits event loop");
    CHECK(wglGetCurrentContext() != NULL && IsWindow(window), "close preserves surface for output cleanup");
    stop_glx3_stuff();
    CHECK(!wglGetCurrentContext() && !glx3_checkEvents(), "teardown releases context");

    glx3_set_window_title(NULL);
    CHECK(start_glx3_stuff(64, 64, 0, argc, argv), "recreate context");
    CHECK(!glx3_should_close() && glx3_checkEvents(), "new context resets close state");
    window = WindowFromDC(wglGetCurrentDC());
    GetWindowTextW(window, title, 256);
    CHECK(wcscmp(title, L"SAM3DBody-cpp OpenGL3.x+ Visualization") == 0, "reset default title");
    SendMessageW(window, WM_KEYDOWN, VK_ESCAPE, 0);
    CHECK(glx3_should_close() && !glx3_checkEvents(), "Escape exits event loop");
    stop_glx3_stuff();
    stop_glx3_stuff();
    {
        /* Exercise legacy Windows main(char **) input when the system code
         * page can represent the title without replacement characters. */
        char ansi_title[256];
        BOOL used_default = FALSE;
        if (GetACP() != CP_UTF8 && WideCharToMultiByte(CP_ACP, 0, L"WGL \x6D4B\x8BD5", -1,
                ansi_title, sizeof(ansi_title), NULL, &used_default) && !used_default) {
            glx3_set_window_title(ansi_title);
            CHECK(start_glx3_stuff(64, 64, 0, argc, argv), "ANSI title context");
            window = WindowFromDC(wglGetCurrentDC());
            GetWindowTextW(window, title, 256);
            CHECK(wcscmp(title, L"WGL \x6D4B\x8BD5") == 0, "native ANSI CLI title");
            stop_glx3_stuff();
        }
    }
    printf("PASS: WGL 3.3 shaders, readback, title, resize, close and recreation\n");
    return 0;
}
