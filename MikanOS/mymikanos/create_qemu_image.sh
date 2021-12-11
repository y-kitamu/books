#!/bin/bash

src_dir=$(dirname ${BASH_SOURCE:-$0})
source ${src_dir}/build_kernel.sh
source ${src_dir}/build_loader.sh

rm disk.img
qemu-img create -f raw disk.img 200M
mkfs.fat -n "MIKAN OS" -s 2 -f 2 -R 32 -F 32 disk.img

mkdir -p mnt
sudo mount -o loop disk.img mnt
sudo mkdir -p mnt/EFI/BOOT
sudo cp ${target} mnt/EFI/BOOT/BOOTX64.EFI
sudo cp ${src_dir}/kernel/kernel.elf mnt/
sudo umount mnt

qemu-system-x86_64 \
    -m 1G \
    -drive if=pflash,format=raw,file=$HOME/work/Learning/Book/MikanOS/mikanos-build/devenv/OVMF_CODE.fd \
    -drive if=pflash,format=raw,file=$HOME/work/Learning/Book/MikanOS/mikanos-build/devenv/OVMF_VARS.fd \
    -drive if=ide,index=0,media=disk,format=raw,file=./disk.img \
    -device nec-usb-xhci,id=xhci \
    -device usb-mouse -device usb-kbd \
    -monitor stdio
