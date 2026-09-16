#include "../src/GraphicsEngine/System/portable_getline.h"
#include <stdlib.h>
#include <string.h>

/* BVH motion rows can exceed a fixed input buffer and omit a final newline. */
int main(void)
{
    FILE *input = tmpfile();
    char *line = NULL;
    size_t capacity = 0;
    int result = 1;
    if (!input) return 1;
    fputs("\n", input);
    for (int i = 0; i < 8192; ++i) fputc('x', input);
    fputs("\nlast", input);
    rewind(input);
    if (fsb_getline(&line, &capacity, input) != 1 || strcmp(line, "\n")) goto done;
    if (fsb_getline(&line, &capacity, input) != 8193 || line[8192] != '\n' || line[8193]) goto done;
    for (int i = 0; i < 8192; ++i) if (line[i] != 'x') goto done;
    if (fsb_getline(&line, &capacity, input) != 4 || strcmp(line, "last")) goto done;
    if (fsb_getline(&line, &capacity, input) != -1) goto done;
    result = 0;
done:
    free(line);
    fclose(input);
    if (result) fprintf(stderr, "FAIL: growing BVH line reader\n");
    else printf("PASS: empty line, long BVH row, final line and EOF\n");
    return result;
}
