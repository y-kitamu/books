global main

main:
    push 0x000a6948
    mov rax, 0x4
    mov rbx, 0x1
    mov rcx, rsp
    mov rdx, 0x4
    int 0x80
    add rsp, 0x4
