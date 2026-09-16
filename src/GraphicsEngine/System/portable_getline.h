#ifndef FSB_PORTABLE_GETLINE_H
#define FSB_PORTABLE_GETLINE_H

#include <stddef.h>
#include <stdio.h>

#ifdef _MSC_VER
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>

/* getline's growing buffer and final unterminated line semantics, for the
 * MSVC CRT. Use a private name instead of introducing POSIX types globally. */
static ptrdiff_t fsb_getline(char **line, size_t *capacity, FILE *stream)
{
    size_t length = 0;
    int ch;
    if (!line || !capacity || !stream) {
        errno = EINVAL;
        return -1;
    }
    if (!*line) *capacity = 0;
    while ((ch = fgetc(stream)) != EOF) {
        if (length + 1 >= *capacity) {
            size_t next = *capacity ? *capacity * 2 : 256;
            char *grown;
            if (next <= *capacity || next > PTRDIFF_MAX) {
                errno = ENOMEM;
                return -1;
            }
            grown = (char *)realloc(*line, next);
            if (!grown) return -1;
            *line = grown;
            *capacity = next;
        }
        (*line)[length++] = (char)ch;
        if (ch == '\n') break;
    }
    if (length == 0 || ferror(stream)) return -1;
    (*line)[length] = '\0';
    return (ptrdiff_t)length;
}
#else
#define fsb_getline getline
#endif

#endif
