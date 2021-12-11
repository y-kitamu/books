    global main

main:
    mov rax, 0x5
    cmp rax, 0x3
    jz true
    jnz false
true:
    mov rax, 0x1
    jmp finish
false:
    mov rax, 0x1
finish:
