#!/bin/bash

src_dir=$(dirname ${BASH_SOURCE:-$0})
source ${src_dir}/build_kernel.sh
source ${src_dir}/build_loader.sh

device=${1:-sdc1}

sudo umount /dev/${device}
sudo mkfs.fat /dev/${device}
sudo mkdir -p /mnt/usbmem
sudo mount /dev/${device} /mnt/usbmem
sudo mkdir -p /mnt/usbmem/EFI/BOOT
sudo cp ${target} /mnt/usbmem/EFI/BOOT/BOOTX64.EFI
sudo cp ${src_dir}/kernel/kernel.elf /mnt/usbmem/
sudo umount /mnt/usbmem

cd ${cur_dir}
