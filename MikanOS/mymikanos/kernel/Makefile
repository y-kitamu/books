TARGET = kernel.elf
OBJS = main.o graphics.o mouse.o font.o hankaku.o newlib_support.o console.o \
       pci.o asmfunc.o libcxx_support.o logger.o interrupt.o segment.o paging.o memory_manager.o \
       window.o layer.o timer.o frame_buffer.o acpi.o keyboard.o task.o \
       usb/memory.o usb/device.o usb/xhci/ring.o usb/xhci/trb.o usb/xhci/xhci.o \
       usb/xhci/port.o usb/xhci/device.o usb/xhci/devmgr.o usb/xhci/registers.o \
       usb/classdriver/base.o usb/classdriver/hid.o usb/classdriver/keyboard.o \
       usb/classdriver/mouse.o

DEPENDS = $(join $(dir $(OBJS)),$(addprefix .,$(notdir $(OBJS:.o=.d))))
# LIBS += libc++.a libc++abi.a libc++experimental.a

PROJECTROOT = /home/kitamura/work/Learning/Book/MikanOS
BASEDIR = $(PROJECTROOT)/mikanos-build/devenv/x86_64-elf
EDKDIR = $(PROJECTROOT)/edk2
LDPATH = $(PROJECTROOT)/mikanos-build/devenv/x86_64-elf/lib/

CFLAGS += -O2 -Wall -g --target=x86_64-elf -ffreestanding -mno-red-zone -fno-exceptions
CFLAGS   += -O2 -Wall -g --target=x86_64-elf -ffreestanding -mno-red-zone
CXXFLAGS += -O2 -Wall -g --target=x86_64-elf -ffreestanding -mno-red-zone \
            -fno-exceptions -fno-rtti -std=c++17
CPPFLAGS += -I$(BASEDIR)/include/c++/v1 \
	-I$(BASEDIR)/include \
	-I$(BASEDIR)/include/freetype2 \
	-I$(EDKDIR)/MdePkg/Include \
	-I$(EDKDIR)/MdePkg/Include/X64 \
	-I$(PROJECTROOT)/mymikanos/kernel \
	-nostdlibinc -D__ELF__ -D_LDBL_EQ_DBL -D_GNU_SOURCE -D_POSIX_TIMERS \
	-DEFIAPI='__attribute__((ms_abi))'
LDFLAGS += --entry KernelMain -z norelro --image-base 0x100000 --static
LDFLAGS += -L$(LDPATH)

.PHONY: all
all: $(TARGET)

.PHONY: clean
clean:
	rm -rf *.o

kernel.elf: $(OBJS) Makefile
	ld.lld $(LDFLAGS) -o kernel.elf $(OBJS) -lc -lc++ -lc++abi

%.o: %.cpp Makefile
	clang++ $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

.%.d: %.cpp
	clang++ $(CPPFLAGS) $(CXXFLAGS) -MM $< > $@
	$(eval OBJ = $(<:.cpp=.o))
	sed --in-place 's|$(notdir $(OBJ))|$(OBJ)|' $@

%.o: %.c Makefile
	clang $(CPPFLAGS) $(CFLAGS) -c $< -o $@

.%.d: %.c
	clang $(CPPFLAGS) $(CFLAGS) -MM $< > $@
	$(eval OBJ = $(<:.c=.o))
	sed --in-place 's|$(notdir $(OBJ))|$(OBJ)|' $@

%.o: %.asm Makefile
	nasm -f elf64 -o $@ $<

hankaku.bin: hankaku.txt
	/home/kitamura/work/Learning/Book/MikanOS/mikanos/tools/makefont.py -o $@ $<

hankaku.o: hankaku.bin
	objcopy -I binary -O elf64-x86-64 -B i386:x86-64 $< $@

.%.d: %.bin
	touch $@

.PHONY: depends
depends:
	$(MAKE) $(DEPENDS)

-include $(DEPENDS)
