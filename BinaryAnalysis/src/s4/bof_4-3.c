#include <stdio.h>
#include <string.h>


void vuln(char* str) {
    char str2[] = "beefbeef";
    char overflowme[16];
    printf("Enter text\n");
    scanf("%s", overflowme);
    printf("%s\n", str2);
    /* printf("%s\n", overflowme); */
    if (strcmp(str, str2) == 0) {
        printf("hacked!\n");
    } else {
        printf("failed!\n");
    }
}

int main() {
    char string[] = "fishfish";
    vuln(string);
}
