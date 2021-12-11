#include <stdio.h>


int pwn() {
    printf("hacked!\n");
}


int vuln() {
    char overflowme[48];
    scanf("%[^\n]", overflowme);
}


int main() {
    vuln();
    printf("failed!\n");
    return 0;
}
